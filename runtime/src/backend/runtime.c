#define _GNU_SOURCE
#include "../core/internal.h"
#include <dlfcn.h>
#include <onnxruntime_c_api.h>

struct buffers {
	int width;
	int height;
	OrtValue *host_input;
	OrtValue *host_output;
	OrtValue *device_input;
	OrtValue *device_output;
};

struct strata_context {
	const OrtApi *api;
	OrtEnv *env;
	OrtSessionOptions *options;
	OrtSession *session;
	OrtSession *fov;
	OrtIoBinding *binding;
	OrtMemoryInfo *cpu;
	const OrtEpDevice *device;
	struct buffers buffers;
	char *model_path;
	char *fov_path;
	const char *provider;
	char fallback[1024];
};

static int ort_failed(const OrtApi *api, OrtStatus *status,
		      struct strata_error *err)
{
	if (!status)
		return 0;
	strata_fail(err, "ONNX Runtime: %s", api->GetErrorMessage(status));
	api->ReleaseStatus(status);
	return -1;
}

static void ORT_API_CALL quiet_log(void *user, OrtLoggingLevel severity,
				   const char *category, const char *id,
				   const char *location, const char *message)
{
	(void)user;
	(void)severity;
	(void)category;
	(void)id;
	(void)location;
	(void)message;
}

static char *cuda_library(void)
{
	const char *name = "libonnxruntime_providers_cuda.so";
	Dl_info info;
	const char *slash;
	char *path;
	size_t directory;

	if (!dladdr(OrtGetApiBase(), &info) || !info.dli_fname)
		return NULL;
	slash = strrchr(info.dli_fname, '/');
	directory = slash ? (size_t)(slash - info.dli_fname) + 1 : 0;
	path = malloc(directory + strlen(name) + 1);
	if (!path)
		return NULL;
	memcpy(path, info.dli_fname, directory);
	strcpy(path + directory, name);
	return path;
}

static const OrtEpDevice *find_cuda(strata_context *ctx, int device_id)
{
	const OrtEpDevice *const *devices;
	size_t count;
	int id;

	if (ort_failed(ctx->api,
		       ctx->api->GetEpDevices(ctx->env, &devices, &count),
		       NULL))
		return NULL;
	for (size_t i = 0; i < count; ++i) {
		const OrtMemoryInfo *memory = ctx->api->EpDevice_MemoryInfo(
			devices[i], OrtDeviceMemoryType_DEFAULT);

		if (strcmp(ctx->api->EpDevice_EpName(devices[i]),
			   "CUDAExecutionProvider") ||
		    !memory ||
		    ort_failed(ctx->api, ctx->api->MemoryInfoGetId(memory, &id),
			       NULL))
			continue;
		if (id == device_id)
			return devices[i];
	}
	return NULL;
}

static int append_cuda(strata_context *ctx, int device_id,
		       struct strata_error *err)
{
	const char *keys[] = { "use_tf32" };
	const char *values[] = { "0" };
	char *library = cuda_library();
	OrtStatus *status;
	void *pin;

	if (!library)
		return strata_fail(err, "cannot locate the CUDA provider");
	status = ctx->api->RegisterExecutionProviderLibrary(ctx->env, "cuda",
							    library);
	/* ORT unloads EP libraries with the env, leaking CUDA's static state. */
	pin = dlopen(library, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
	if (pin)
		dlclose(pin);
	free(library);
	ctx->device = find_cuda(ctx, device_id);
	if (!ctx->device)
		return status ?
			       ort_failed(ctx->api, status, err) :
			       strata_fail(err, "no CUDA device %d", device_id);
	if (status)
		ctx->api->ReleaseStatus(status);
	return ort_failed(ctx->api,
			  ctx->api->SessionOptionsAppendExecutionProvider_V2(
				  ctx->options, ctx->env, &ctx->device, 1, keys,
				  values, 1),
			  err);
}

static int select_provider(strata_context *ctx,
			   const struct strata_options *options,
			   struct strata_error *err)
{
	const char *name = options->provider ? options->provider : "auto";
	struct strata_error reason = { { 0 } };

	ctx->provider = "cpu";
	if (!strcmp(name, "cpu"))
		return 0;
	if (strcmp(name, "auto") && strcmp(name, "cuda"))
		return strata_fail(err, "unknown provider: %s", name);
	if (!append_cuda(ctx, options->device_id, &reason)) {
		ctx->provider = "cuda";
		return 0;
	}
	ctx->device = NULL;
	if (!strcmp(name, "cuda"))
		return strata_fail(err, "%s", reason.message);
	snprintf(ctx->fallback, sizeof(ctx->fallback), "%s", reason.message);
	return 0;
}

static int tensor_info(strata_context *ctx, OrtSession *session, int input,
		       char **name, OrtTypeInfo **type_info,
		       OrtAllocator *allocator, struct strata_error *err)
{
	const OrtApi *api = ctx->api;
	OrtStatus *status;

	status = input ? api->SessionGetInputName(session, 0, allocator, name) :
			 api->SessionGetOutputName(session, 0, allocator, name);
	if (ort_failed(api, status, err))
		return -1;
	status = input ? api->SessionGetInputTypeInfo(session, 0, type_info) :
			 api->SessionGetOutputTypeInfo(session, 0, type_info);
	return ort_failed(api, status, err);
}

static int shape_matches(strata_context *ctx,
			 const OrtTensorTypeAndShapeInfo *info,
			 const int64_t *expected, size_t rank, int *matches,
			 struct strata_error *err)
{
	const OrtApi *api = ctx->api;
	ONNXTensorElementDataType type;
	int64_t shape[4];
	size_t dimensions;

	*matches = 0;
	if (ort_failed(api, api->GetTensorElementType(info, &type), err) ||
	    ort_failed(api, api->GetDimensionsCount(info, &dimensions), err))
		return -1;
	if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || dimensions != rank)
		return 0;
	if (ort_failed(api, api->GetDimensions(info, shape, rank), err))
		return -1;
	for (size_t i = 0; i < rank; ++i) {
		if (expected[i] < 0 ? shape[i] >= 0 : shape[i] != expected[i])
			return 0;
	}
	*matches = 1;
	return 0;
}

static int tensor_contract(strata_context *ctx, OrtSession *session, int input,
			   const char *expected_name, const int64_t *expected,
			   size_t rank, struct strata_error *err)
{
	const OrtApi *api = ctx->api;
	const OrtTensorTypeAndShapeInfo *info = NULL;
	OrtTypeInfo *type_info = NULL;
	OrtAllocator *allocator;
	char *name = NULL;
	int matches = 0, ret = -1;

	if (ort_failed(api, api->GetAllocatorWithDefaultOptions(&allocator),
		       err) ||
	    tensor_info(ctx, session, input, &name, &type_info, allocator,
			err) ||
	    ort_failed(api, api->CastTypeInfoToTensorInfo(type_info, &info),
		       err))
		goto out;
	if (info && !strcmp(name, expected_name) &&
	    shape_matches(ctx, info, expected, rank, &matches, err))
		goto out;
	ret = matches ? 0 :
			strata_fail(err, "ONNX tensor contract mismatch: %s",
				    expected_name);
out:
	if (name)
		allocator->Free(allocator, name);
	if (type_info)
		api->ReleaseTypeInfo(type_info);
	return ret;
}

static int session_contract(strata_context *ctx, OrtSession *session, int fov,
			    struct strata_error *err)
{
	const int64_t image[] = { 1, 3, -1, -1 };
	const int64_t depth[] = { 1, 1, -1, -1 };
	const int64_t angles[] = { 1, 2 };
	size_t inputs, outputs;

	if (ort_failed(ctx->api,
		       ctx->api->SessionGetInputCount(session, &inputs), err) ||
	    ort_failed(ctx->api,
		       ctx->api->SessionGetOutputCount(session, &outputs), err))
		return -1;
	if (inputs != 1 || outputs != 1)
		return strata_fail(
			err,
			"ONNX tensor contract requires one input and one output");
	if (tensor_contract(ctx, session, 1, STRATA_INPUT_NAME, image, 4, err))
		return -1;
	return tensor_contract(ctx, session, 0,
			       fov ? STRATA_FOV_OUTPUT_NAME :
				     STRATA_OUTPUT_NAME,
			       fov ? angles : depth, fov ? 2u : 4u, err);
}

static int load_session(strata_context *ctx, const char *path,
			OrtSession **session, int fov, struct strata_error *err)
{
	if (ort_failed(ctx->api,
		       ctx->api->CreateSession(ctx->env, path, ctx->options,
					       session),
		       err))
		return -1;
	if (session_contract(ctx, *session, fov, err)) {
		ctx->api->ReleaseSession(*session);
		*session = NULL;
		return -1;
	}
	return 0;
}

static int create_context(const struct strata_options *options,
			  strata_context *ctx, struct strata_error *err)
{
	const OrtApi *api = ctx->api;

	if (ort_failed(api,
		       api->CreateEnvWithCustomLogger(quiet_log, NULL,
						      ORT_LOGGING_LEVEL_WARNING,
						      "strata", &ctx->env),
		       err) ||
	    ort_failed(api, api->CreateSessionOptions(&ctx->options), err) ||
	    ort_failed(api,
		       api->CreateCpuMemoryInfo(OrtArenaAllocator,
						OrtMemTypeDefault, &ctx->cpu),
		       err) ||
	    select_provider(ctx, options, err))
		return -1;
	ctx->model_path = malloc(strlen(options->model_path) + 1);
	if (ctx->model_path)
		strcpy(ctx->model_path, options->model_path);
	ctx->fov_path =
		strata_sibling_path(options->model_path, STRATA_FOV_SUFFIX);
	if (!ctx->model_path || !ctx->fov_path)
		return strata_fail(err, "out of memory");
	return 0;
}

int strata_open(const struct strata_options *options, strata_context **out,
		struct strata_error *err)
{
	strata_context *ctx;

	if (!out)
		return strata_fail(err, "missing context output");
	*out = NULL;
	if (!options || !options->model_path || options->device_id < 0)
		return strata_fail(err, "invalid session options");
	if (strata_check_manifest(options->model_path, err))
		return -1;
	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return strata_fail(err, "out of memory");
	ctx->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (!ctx->api) {
		free(ctx);
		return strata_fail(err, "ONNX Runtime API version mismatch");
	}
	if (create_context(options, ctx, err) ||
	    load_session(ctx, ctx->model_path, &ctx->session, 0, err)) {
		strata_close(ctx);
		return -1;
	}
	*out = ctx;
	return 0;
}

static void release_value(const OrtApi *api, OrtValue **value)
{
	if (*value)
		api->ReleaseValue(*value);
	*value = NULL;
}

static void release_buffers(strata_context *ctx)
{
	if (ctx->binding)
		ctx->api->ReleaseIoBinding(ctx->binding);
	ctx->binding = NULL;
	release_value(ctx->api, &ctx->buffers.host_input);
	release_value(ctx->api, &ctx->buffers.host_output);
	release_value(ctx->api, &ctx->buffers.device_input);
	release_value(ctx->api, &ctx->buffers.device_output);
	ctx->buffers.width = 0;
	ctx->buffers.height = 0;
}

void strata_unload(strata_context *ctx)
{
	if (!ctx)
		return;
	release_buffers(ctx);
	if (ctx->fov)
		ctx->api->ReleaseSession(ctx->fov);
	if (ctx->session)
		ctx->api->ReleaseSession(ctx->session);
	ctx->fov = NULL;
	ctx->session = NULL;
}

void strata_close(strata_context *ctx)
{
	if (!ctx)
		return;
	strata_unload(ctx);
	if (ctx->cpu)
		ctx->api->ReleaseMemoryInfo(ctx->cpu);
	if (ctx->options)
		ctx->api->ReleaseSessionOptions(ctx->options);
	if (ctx->env)
		ctx->api->ReleaseEnv(ctx->env);
	free(ctx->model_path);
	free(ctx->fov_path);
	free(ctx);
}

const char *strata_provider(const strata_context *ctx)
{
	return ctx->provider;
}

const char *strata_fallback_reason(const strata_context *ctx)
{
	return ctx->fallback;
}

static int allocator(strata_context *ctx, OrtDeviceMemoryType type,
		     OrtAllocator **out, struct strata_error *err)
{
	const OrtMemoryInfo *memory;

	if (!ctx->device)
		return ort_failed(ctx->api,
				  ctx->api->GetAllocatorWithDefaultOptions(out),
				  err);
	memory = ctx->api->EpDevice_MemoryInfo(ctx->device, type);
	if (ort_failed(ctx->api,
		       ctx->api->GetSharedAllocator(ctx->env, memory, out),
		       err))
		return -1;
	return *out ? 0 : strata_fail(err, "no CUDA allocator");
}

static int create_tensor(strata_context *ctx, OrtDeviceMemoryType type,
			 const int64_t *shape, OrtValue **value,
			 struct strata_error *err)
{
	OrtAllocator *memory;

	if (allocator(ctx, type, &memory, err))
		return -1;
	return ort_failed(ctx->api,
			  ctx->api->CreateTensorAsOrtValue(
				  memory, shape, 4,
				  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, value),
			  err);
}

static int bind_buffers(strata_context *ctx, struct strata_error *err)
{
	struct buffers *b = &ctx->buffers;
	OrtValue *input = ctx->device ? b->device_input : b->host_input;
	OrtValue *output = ctx->device ? b->device_output : b->host_output;

	if (ort_failed(ctx->api,
		       ctx->api->CreateIoBinding(ctx->session, &ctx->binding),
		       err) ||
	    ort_failed(ctx->api,
		       ctx->api->BindInput(ctx->binding, STRATA_INPUT_NAME,
					   input),
		       err))
		return -1;
	return ort_failed(ctx->api,
			  ctx->api->BindOutput(ctx->binding, STRATA_OUTPUT_NAME,
					       output),
			  err);
}

static int ensure_buffers(strata_context *ctx, int pw, int ph,
			  struct strata_error *err)
{
	const int64_t input[] = { 1, 3, ph, pw };
	const int64_t output[] = { 1, 1, ph, pw };
	OrtDeviceMemoryType host = ctx->device ?
					   OrtDeviceMemoryType_HOST_ACCESSIBLE :
					   OrtDeviceMemoryType_DEFAULT;
	struct buffers *b = &ctx->buffers;

	if (b->width == pw && b->height == ph)
		return 0;
	release_buffers(ctx);
	if (create_tensor(ctx, host, input, &b->host_input, err) ||
	    create_tensor(ctx, host, output, &b->host_output, err) ||
	    (ctx->device && (create_tensor(ctx, OrtDeviceMemoryType_DEFAULT,
					   input, &b->device_input, err) ||
			     create_tensor(ctx, OrtDeviceMemoryType_DEFAULT,
					   output, &b->device_output, err))) ||
	    bind_buffers(ctx, err)) {
		release_buffers(ctx);
		return -1;
	}
	b->width = pw;
	b->height = ph;
	return 0;
}

static float *tensor_data(strata_context *ctx, OrtValue *value)
{
	void *data = NULL;

	if (ort_failed(ctx->api, ctx->api->GetTensorMutableData(value, &data),
		       NULL))
		return NULL;
	return data;
}

static int estimate_focal(strata_context *ctx,
			  const struct strata_tensor *padded, int w, int h,
			  float *focal, struct strata_error *err)
{
	struct strata_tensor crop = { 0 };
	OrtValue *input = NULL, *output = NULL;
	const char *input_name = STRATA_INPUT_NAME;
	const char *output_name = STRATA_FOV_OUTPUT_NAME;
	int64_t shape[4] = { 1, 3, 0, 0 };
	float *angles;
	int ret = -1;

	if ((!ctx->fov &&
	     load_session(ctx, ctx->fov_path, &ctx->fov, 1, err)) ||
	    strata_fov_input(padded, w, h, &crop, err))
		goto out;
	shape[2] = crop.height;
	shape[3] = crop.width;
	if (ort_failed(ctx->api,
		       ctx->api->CreateTensorWithDataAsOrtValue(
			       ctx->cpu, crop.data,
			       (size_t)3 * (size_t)crop.width *
				       (size_t)crop.height * sizeof(float),
			       shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			       &input),
		       err) ||
	    ort_failed(ctx->api,
		       ctx->api->Run(ctx->fov, NULL, &input_name,
				     (const OrtValue *const *)&input, 1,
				     &output_name, 1, &output),
		       err))
		goto out;
	angles = tensor_data(ctx, output);
	if (!angles) {
		strata_fail(err, "focal output unavailable");
		goto out;
	}
	ret = strata_focal(angles, w, h, focal, err);
out:
	release_value(ctx->api, &input);
	release_value(ctx->api, &output);
	free(crop.data);
	return ret;
}

static int copy_tensor(strata_context *ctx, OrtValue *source, OrtValue *target,
		       double *ms, struct strata_error *err)
{
	double start = strata_milliseconds();
	int ret = 0;

	if (ctx->device)
		ret = ort_failed(
			ctx->api,
			ctx->api->CopyTensors(ctx->env,
					      (const OrtValue *const *)&source,
					      &target, NULL, 1),
			err);
	*ms = strata_milliseconds() - start;
	return ret;
}

static int focal_length(strata_context *ctx, const struct strata_frame *frame,
			const struct strata_tensor *padded,
			struct strata_result *result, struct strata_error *err)
{
	struct strata_camera *camera = &result->frame.camera;
	int w = frame->image->width, h = frame->image->height;
	double start = strata_milliseconds();

	*camera = frame->camera;
	if (camera->focal_source == STRATA_FOCAL_ESTIMATED) {
		if (estimate_focal(ctx, padded, w, h, &camera->fx_px, err))
			return -1;
		camera->fy_px = camera->fx_px;
		camera->cx_px = (float)w / 2;
		camera->cy_px = (float)h / 2;
		if (strata_camera_validate(camera, 1, err))
			return -1;
	}
	result->timings.focal_ms = strata_milliseconds() - start;
	return 0;
}

static int infer(strata_context *ctx, const struct strata_frame *frame,
		 struct strata_result *result, struct strata_error *err)
{
	struct buffers *b = &ctx->buffers;
	struct strata_timings *timings = &result->timings;
	struct strata_tensor padded;
	float *prediction;
	double start;

	if (strata_padded_size(frame->image->width, frame->image->height,
			       &padded.width, &padded.height, err) ||
	    ensure_buffers(ctx, padded.width, padded.height, err))
		return -1;
	padded.data = tensor_data(ctx, b->host_input);
	prediction = tensor_data(ctx, b->host_output);
	if (!padded.data || !prediction)
		return strata_fail(err, "inference buffers unavailable");
	start = strata_milliseconds();
	result->nonfinite_input_pixels =
		strata_prepare(frame->image, frame->exposure_stops, &padded);
	timings->prepare_ms = strata_milliseconds() - start;
	if (focal_length(ctx, frame, &padded, result, err) ||
	    copy_tensor(ctx, b->host_input, b->device_input,
			&timings->upload_ms, err))
		return -1;
	start = strata_milliseconds();
	if (ort_failed(ctx->api,
		       ctx->api->RunWithBinding(ctx->session, NULL,
						ctx->binding),
		       err))
		return -1;
	timings->run_ms = strata_milliseconds() - start;
	if (copy_tensor(ctx, b->device_output, b->host_output,
			&timings->download_ms, err))
		return -1;
	start = strata_milliseconds();
	if (strata_postprocess(prediction, padded.width, result, err))
		return -1;
	timings->postprocess_ms = strata_milliseconds() - start;
	return 0;
}

int strata_run(strata_context *ctx, const struct strata_frame *frame,
	       struct strata_result *result, struct strata_error *err)
{
	if (!result)
		return strata_fail(err, "missing result output");
	strata_result_free(result);
	if (!ctx || !frame)
		return strata_fail(err, "invalid run arguments");
	if (!frame->image || !frame->image->rgba ||
	    !isfinite(frame->exposure_stops) ||
	    !isfinite(frame->timestamp_seconds))
		return strata_fail(err, "invalid frame");
	if (strata_camera_validate(&frame->camera, 0, err))
		return -1;
	if (!ctx->session &&
	    (strata_check_manifest(ctx->model_path, err) ||
	     load_session(ctx, ctx->model_path, &ctx->session, 0, err)))
		return -1;
	result->width = frame->image->width;
	result->height = frame->image->height;
	result->frame.index = frame->index;
	result->frame.timestamp_seconds = frame->timestamp_seconds;
	result->frame.linear = frame->image->linear;
	if (infer(ctx, frame, result, err)) {
		strata_result_free(result);
		return -1;
	}
	return 0;
}
