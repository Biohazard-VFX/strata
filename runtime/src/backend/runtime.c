#define _GNU_SOURCE
#include "../core/internal.h"
#include <dlfcn.h>
#include <onnxruntime_c_api.h>
#include <openssl/evp.h>

struct buffers {
	int width;
	int height;
	OrtValue *host_input;
	OrtValue *host_output;
	OrtValue *device_input;
	OrtValue *device_output;
	OrtValue *host_features;
	OrtValue *device_features;
};

struct strata_context {
	const OrtApi *api;
	OrtEnv *env;
	OrtSessionOptions *options;
	OrtSession *session;
	OrtSession *fov;
	OrtSession *head;
	char *head_path;
	char *feature_path;
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
		       size_t index, char **name, OrtTypeInfo **type_info,
		       OrtAllocator *allocator, struct strata_error *err)
{
	const OrtApi *api = ctx->api;
	OrtStatus *status;

	status = input ? api->SessionGetInputName(session, index, allocator,
						  name) :
			 api->SessionGetOutputName(session, index, allocator,
						   name);
	if (ort_failed(api, status, err))
		return -1;
	status = input ? api->SessionGetInputTypeInfo(session, index,
						      type_info) :
			 api->SessionGetOutputTypeInfo(session, index,
						       type_info);
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
			   size_t index, const char *expected_name,
			   const int64_t *expected, size_t rank,
			   struct strata_error *err)
{
	const OrtApi *api = ctx->api;
	const OrtTensorTypeAndShapeInfo *info = NULL;
	OrtTypeInfo *type_info = NULL;
	OrtAllocator *allocator;
	char *name = NULL;
	int matches = 0, ret = -1;

	if (ort_failed(api, api->GetAllocatorWithDefaultOptions(&allocator),
		       err) ||
	    tensor_info(ctx, session, input, index, &name, &type_info,
			allocator, err) ||
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
	if (inputs != 1 || outputs != (fov == 2 ? 2u : 1u))
		return strata_fail(
			err,
			"ONNX tensor contract requires one input and one output");
	if (tensor_contract(ctx, session, 1, 0, STRATA_INPUT_NAME, image, 4,
			    err))
		return -1;
	return tensor_contract(
		ctx, session, 0, 0,
		fov == 1 ? STRATA_FOV_OUTPUT_NAME : STRATA_OUTPUT_NAME,
		fov == 1 ? angles : depth, fov == 1 ? 2u : 4u, err);
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
	ctx->fov_path = strata_fov_path(options->model_path);
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
	release_value(ctx->api, &ctx->buffers.host_features);
	release_value(ctx->api, &ctx->buffers.device_features);
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
	if (ctx->head)
		ctx->api->ReleaseSession(ctx->head);
	ctx->head = NULL;
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
	free(ctx->head_path);
	free(ctx->feature_path);
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
	if (ctx->head_path &&
	    ort_failed(ctx->api,
		       ctx->api->BindOutput(ctx->binding, "strata_features",
					    ctx->device ? b->device_features :
							  b->host_features),
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
	const int64_t features[] = { 1, 128, ph, pw };
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
	    (ctx->head_path &&
	     (create_tensor(ctx, host, features, &b->host_features, err) ||
	      (ctx->device &&
	       create_tensor(ctx, OrtDeviceMemoryType_DEFAULT, features,
			     &b->device_features, err)))) ||
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
	if (ctx->head_path) {
		double feature_ms;
		if (copy_tensor(ctx, b->device_features, b->host_features,
				&feature_ms, err))
			return -1;
		timings->download_ms += feature_ms;
	}
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
	     load_session(ctx,
			  ctx->feature_path ? ctx->feature_path :
					      ctx->model_path,
			  &ctx->session, ctx->feature_path ? 2 : 0, err)))
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

static int metadata_matches(strata_context *ctx, const char *key,
			    const char *expected, struct strata_error *err)
{
	OrtModelMetadata *metadata = NULL;
	OrtAllocator *memory;
	char *value = NULL;
	int ret = -1;

	if (ort_failed(ctx->api,
		       ctx->api->GetAllocatorWithDefaultOptions(&memory),
		       err) ||
	    ort_failed(ctx->api,
		       ctx->api->SessionGetModelMetadata(ctx->head, &metadata),
		       err) ||
	    ort_failed(ctx->api,
		       ctx->api->ModelMetadataLookupCustomMetadataMap(
			       metadata, memory, key, &value),
		       err))
		goto out;
	ret = value && !strcmp(value, expected) ?
		      0 :
		      strata_fail(err, "head metadata mismatch: %s", key);
out:
	if (value)
		memory->Free(memory, value);
	if (metadata)
		ctx->api->ReleaseModelMetadata(metadata);
	return ret;
}

static int model_digest(const char *path, char text[65],
			struct strata_error *err)
{
	FILE *stream = fopen(path, "rb");
	EVP_MD_CTX *digest = EVP_MD_CTX_new();
	unsigned char buffer[65536], hash[32];
	size_t count;
	unsigned int length;
	int ret = -1;

	if (!stream || !digest ||
	    !EVP_DigestInit_ex(digest, EVP_sha256(), NULL))
		goto out;
	while ((count = fread(buffer, 1, sizeof(buffer), stream)) != 0)
		if (!EVP_DigestUpdate(digest, buffer, count))
			goto out;
	if (ferror(stream) || !EVP_DigestFinal_ex(digest, hash, &length) ||
	    length != 32)
		goto out;
	for (size_t i = 0; i < 32; ++i)
		snprintf(text + 2 * i, 3, "%02x", hash[i]);
	ret = 0;
out:
	if (stream)
		fclose(stream);
	EVP_MD_CTX_free(digest);
	return ret ? strata_fail(err, "cannot hash backbone") : 0;
}

static int head_contract(strata_context *ctx, struct strata_error *err)
{
	const char *inputs[] = { "features", "image", "base" };
	const char *outputs[] = { "inverse_depth", "alpha" };
	const int64_t channels[] = { 128, 3, 1 };
	size_t count;

	if (ort_failed(ctx->api,
		       ctx->api->SessionGetInputCount(ctx->head, &count), err))
		return -1;
	if (count != 3)
		return strata_fail(err, "head requires three inputs");
	if (ort_failed(ctx->api,
		       ctx->api->SessionGetOutputCount(ctx->head, &count), err))
		return -1;
	if (count != 2)
		return strata_fail(err, "head requires two outputs");
	for (size_t i = 0; i < 3; ++i) {
		int64_t shape[] = { 1, channels[i], -1, -1 };
		if (tensor_contract(ctx, ctx->head, 1, i, inputs[i], shape, 4,
				    err))
			return -1;
	}
	for (size_t i = 0; i < 2; ++i) {
		int64_t shape[] = { 1, 4, -1, -1 };
		if (tensor_contract(ctx, ctx->head, 0, i, outputs[i], shape, 4,
				    err))
			return -1;
	}
	return 0;
}

static int load_head(strata_context *ctx, struct strata_error *err)
{
	char digest[65];

	if (ort_failed(ctx->api,
		       ctx->api->CreateSession(ctx->env, ctx->head_path,
					       ctx->options, &ctx->head),
		       err) ||
	    metadata_matches(ctx, "strata.objective", "coverage-simplex-bce-v1",
			     err) ||
	    metadata_matches(ctx, "strata.layers", "4", err) ||
	    metadata_matches(ctx, "strata.colour",
			     "plate-premultiplied-conditional-alpha", err) ||
	    model_digest(ctx->model_path, digest, err) ||
	    metadata_matches(ctx, "strata.backbone_sha256", digest, err) ||
	    model_digest(ctx->feature_path, digest, err) ||
	    metadata_matches(ctx, "strata.feature_backbone_sha256", digest,
			     err) ||
	    head_contract(ctx, err)) {
		if (ctx->head)
			ctx->api->ReleaseSession(ctx->head);
		ctx->head = NULL;
		return -1;
	}
	return 0;
}

int strata_set_head(strata_context *ctx, const char *path,
		    struct strata_error *err)
{
	size_t length;

	if (!ctx || !path)
		return strata_fail(err, "missing head path or context");
	length = strlen(path);
	if (length < 5 || strcmp(path + length - 5, ".onnx"))
		return strata_fail(err, "head path must end in .onnx");
	strata_unload(ctx);
	free(ctx->head_path);
	free(ctx->feature_path);
	ctx->head_path = strdup(path);
	ctx->feature_path = malloc(length + sizeof(".backbone.onnx"));
	if (!ctx->head_path || !ctx->feature_path)
		return strata_fail(err, "out of memory");
	memcpy(ctx->feature_path, path, length - 5);
	strcpy(ctx->feature_path + length - 5, ".backbone.onnx");
	return load_head(ctx, err);
}

struct head_tile {
	int x, y, x0, y0, width, height;
	float *features, *image, *base;
};

static int tile_inputs(strata_context *ctx, const struct strata_frame *frame,
		       const struct strata_result *depth,
		       struct head_tile *tile, struct strata_error *err)
{
	const float *features = tensor_data(ctx, ctx->buffers.host_features);
	size_t padded =
		(size_t)ctx->buffers.width * (size_t)ctx->buffers.height;
	size_t pixels = (size_t)tile->width * (size_t)tile->height;

	tile->features = malloc(pixels * 128 * sizeof(float));
	tile->image = malloc(pixels * 3 * sizeof(float));
	tile->base = malloc(pixels * sizeof(float));
	if (!features || !tile->features || !tile->image || !tile->base)
		return strata_fail(err, "head input allocation failed");
	for (int y = 0; y < tile->height; ++y) {
		for (int x = 0; x < tile->width; ++x) {
			size_t local =
				(size_t)y * (size_t)tile->width + (size_t)x;
			size_t pixel =
				(size_t)(tile->y0 + y) * (size_t)depth->width +
				(size_t)(tile->x0 + x);
			size_t padded_pixel =
				(size_t)(tile->y0 + y) *
					(size_t)ctx->buffers.width +
				(size_t)(tile->x0 + x);
			for (size_t c = 0; c < 128; ++c)
				tile->features[c * pixels + local] =
					features[c * padded + padded_pixel];
			for (size_t c = 0; c < 3; ++c)
				tile->image[c * pixels + local] =
					frame->image->rgba[pixel * 4 + c];
			tile->base[local] = strata_inverse_metres(depth, pixel);
		}
	}
	return 0;
}

static float *head_output(strata_context *ctx, OrtValue *value, int width,
			  int height, struct strata_error *err)
{
	OrtTensorTypeAndShapeInfo *info = NULL;
	const int64_t shape[] = { 1, 4, height, width };
	int matches = 0;

	if (ort_failed(ctx->api, ctx->api->GetTensorTypeAndShape(value, &info),
		       err))
		return NULL;
	if (shape_matches(ctx, info, shape, 4, &matches, err))
		matches = 0;
	ctx->api->ReleaseTensorTypeAndShapeInfo(info);
	if (!matches) {
		strata_fail(err, "head output shape mismatch");
		return NULL;
	}
	return tensor_data(ctx, value);
}

static int run_tile(strata_context *ctx, struct head_tile *tile, float *inverse,
		    float *alpha, int width, int height,
		    struct strata_error *err)
{
	const char *names[] = { "features", "image", "base" };
	const char *outputs[] = { "inverse_depth", "alpha" };
	float *data[] = { tile->features, tile->image, tile->base };
	int64_t channels[] = { 128, 3, 1 };
	OrtValue *inputs[3] = { 0 }, *values[2] = { 0 };
	size_t pixels = (size_t)tile->width * (size_t)tile->height;
	int ret = -1;

	for (size_t i = 0; i < 3; ++i) {
		int64_t shape[] = { 1, channels[i], tile->height, tile->width };
		if (ort_failed(ctx->api,
			       ctx->api->CreateTensorWithDataAsOrtValue(
				       ctx->cpu, data[i],
				       pixels * (size_t)channels[i] *
					       sizeof(float),
				       shape, 4,
				       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
				       &inputs[i]),
			       err))
			goto out;
	}
	if (ort_failed(ctx->api,
		       ctx->api->Run(ctx->head, NULL, names,
				     (const OrtValue *const *)inputs, 3,
				     outputs, 2, values),
		       err))
		goto out;
	for (size_t i = 0; i < 2; ++i) {
		const float *prediction = head_output(
			ctx, values[i], tile->width, tile->height, err);
		float *target = i ? alpha : inverse;
		if (!prediction)
			goto out;
		for (size_t k = 0; k < 4; ++k)
			for (int y = tile->y; y < height && y < tile->y + 256;
			     ++y)
				for (int x = tile->x;
				     x < width && x < tile->x + 256; ++x)
					target[(k * (size_t)height + (size_t)y) *
						       (size_t)width +
					       (size_t)x] = prediction
						[k * pixels +
						 (size_t)(y - tile->y0) *
							 (size_t)tile->width +
						 (size_t)(x - tile->x0)];
	}
	ret = 0;
out:
	for (size_t i = 0; i < 3; ++i)
		release_value(ctx->api, &inputs[i]);
	for (size_t i = 0; i < 2; ++i)
		release_value(ctx->api, &values[i]);
	return ret;
}

static int predict_layers(strata_context *ctx, const struct strata_frame *frame,
			  const struct strata_result *depth, float *inverse,
			  float *alpha, struct strata_error *err)
{
	int width = depth->width, height = depth->height;

	if (!ctx->head && load_head(ctx, err))
		return -1;
	for (int y = 0; y < height; y += 256) {
		for (int x = 0; x < width; x += 256) {
			int x1 = x + 288 < width ? x + 288 : width;
			int y1 = y + 288 < height ? y + 288 : height;
			struct head_tile tile = { .x = x,
						  .y = y,
						  .x0 = x > 32 ? x - 32 : 0,
						  .y0 = y > 32 ? y - 32 : 0 };
			int ret;
			tile.width = x1 - tile.x0;
			tile.height = y1 - tile.y0;
			ret = tile_inputs(ctx, frame, depth, &tile, err) ||
			      run_tile(ctx, &tile, inverse, alpha, width,
				       height, err);
			free(tile.features);
			free(tile.image);
			free(tile.base);
			if (ret)
				return -1;
		}
	}
	return 0;
}

int strata_run_deep(strata_context *ctx, const struct strata_frame *frame,
		    struct strata_result *depth,
		    struct strata_deep_result *deep, struct strata_error *err)
{
	float *inverse = NULL, *alpha = NULL;
	size_t pixels;
	double start;
	int ret = -1;

	if (!deep)
		return strata_fail(err, "missing deep output");
	strata_deep_free(deep);
	if (!depth)
		return strata_fail(err, "missing depth output");
	strata_result_free(depth);
	if (!ctx || !ctx->head_path)
		return strata_fail(err, "layered deep requires a head");
	if (strata_run(ctx, frame, depth, err))
		return -1;
	start = strata_milliseconds();
	pixels = (size_t)depth->width * (size_t)depth->height;
	inverse = malloc(pixels * 4 * sizeof(float));
	alpha = malloc(pixels * 4 * sizeof(float));
	if (!inverse || !alpha) {
		strata_fail(err, "out of memory");
		goto out;
	}
	if (predict_layers(ctx, frame, depth, inverse, alpha, err) ||
	    strata_prediction_to_deep(frame->image, &depth->frame, 4, inverse,
				      alpha, deep, err))
		goto out;
	depth->timings.run_ms += strata_milliseconds() - start;
	ret = 0;
out:
	free(inverse);
	free(alpha);
	if (ret) {
		strata_deep_free(deep);
		strata_result_free(depth);
	}
	return ret;
}
