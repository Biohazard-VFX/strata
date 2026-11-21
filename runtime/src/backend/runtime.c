#include "../core/internal.h"
#include <onnxruntime_c_api.h>
#ifdef _WIN32
#include <windows.h>
#ifdef STRATA_DIRECTML
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_DML,
	       _In_ OrtSessionOptions *options, int device_id);
#endif
#endif

struct strata_context {
	const OrtApi *api;
	OrtEnv *env;
	OrtSessionOptions *options;
	OrtSession *session;
	OrtSession *fov;
	OrtMemoryInfo *memory;
	char *model_path;
	char *fov_path;
	const char *provider;
};

static int ort_error(strata_context *ctx, OrtStatus *status,
		     struct strata_error *err)
{
	if (!status)
		return 0;
	strata_fail(err, "ONNX Runtime: %s", ctx->api->GetErrorMessage(status));
	ctx->api->ReleaseStatus(status);
	return -1;
}

#define ORT(call)                                        \
	do {                                             \
		if (ort_error(ctx, ctx->api->call, err)) \
			goto out;                        \
	} while (0)

static OrtStatus *create_session(strata_context *ctx, const char *path,
				 OrtSession **session)
{
#ifdef _WIN32
	int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1,
				    NULL, 0);
	wchar_t *wide;
	OrtStatus *status;

	if (!n)
		return ctx->api->CreateStatus(ORT_FAIL, "invalid UTF-8 path");
	wide = malloc((size_t)n * sizeof(wchar_t));
	if (!wide)
		return ctx->api->CreateStatus(ORT_FAIL, "out of memory");
	MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide, n);
	status = ctx->api->CreateSession(ctx->env, wide, ctx->options, session);
	free(wide);
	return status;
#else
	return ctx->api->CreateSession(ctx->env, path, ctx->options, session);
#endif
}

static OrtStatus *append_cuda(strata_context *ctx, int device)
{
	OrtCUDAProviderOptionsV2 *options = NULL;
	OrtStatus *status = ctx->api->CreateCUDAProviderOptions(&options);
	char id[32];
	const char *keys[] = { "device_id", "gpu_mem_limit",
			       "arena_extend_strategy",
			       "cudnn_conv_use_max_workspace", "use_tf32" };
	const char *values[] = { id, "13958643712", "kSameAsRequested", "0",
				 "0" };

	if (status)
		return status;
	snprintf(id, sizeof(id), "%d", device);
	status = ctx->api->UpdateCUDAProviderOptions(options, keys, values, 5);
	if (!status)
		status =
			ctx->api->SessionOptionsAppendExecutionProvider_CUDA_V2(
				ctx->options, options);
	ctx->api->ReleaseCUDAProviderOptions(options);
	return status;
}

static OrtStatus *append_tensorrt(strata_context *ctx, int device)
{
	OrtTensorRTProviderOptionsV2 *options = NULL;
	OrtStatus *status = ctx->api->CreateTensorRTProviderOptions(&options);
	char id[32];
	const char *keys[] = { "device_id", "trt_fp16_enable",
			       "trt_max_workspace_size" };
	const char *values[] = { id, "1", "1073741824" };

	if (status)
		return status;
	snprintf(id, sizeof(id), "%d", device);
	status = ctx->api->UpdateTensorRTProviderOptions(options, keys, values,
							 3);
	if (!status)
		status =
			ctx->api->SessionOptionsAppendExecutionProvider_TensorRT_V2(
				ctx->options, options);
	ctx->api->ReleaseTensorRTProviderOptions(options);
	if (!status)
		status = append_cuda(ctx, device);
	return status;
}

static OrtStatus *append_provider(strata_context *ctx, const char *name,
				  int device)
{
	if (!strcmp(name, "cpu"))
		return NULL;
	if (!strcmp(name, "cuda"))
		return append_cuda(ctx, device);
	if (!strcmp(name, "tensorrt"))
		return append_tensorrt(ctx, device);
#ifdef __APPLE__
	if (!strcmp(name, "coreml")) {
		const char *keys[] = { "MLComputeUnits", "ModelFormat" };
		const char *values[] = { "ALL", "MLProgram" };

		return ctx->api->SessionOptionsAppendExecutionProvider(
			ctx->options, "CoreML", keys, values, 2);
	}
#endif
#if defined(_WIN32) && defined(STRATA_DIRECTML)
	if (!strcmp(name, "directml")) {
		OrtStatus *status = ctx->api->DisableMemPattern(ctx->options);

		if (!status)
			status = OrtSessionOptionsAppendExecutionProvider_DML(
				ctx->options, device);
		return status;
	}
#endif
	return ctx->api->CreateStatus(
		ORT_INVALID_ARGUMENT,
		"unknown provider or provider not built on this platform");
}

static int select_provider(strata_context *ctx,
			   const struct strata_options *options,
			   struct strata_error *err)
{
	const char *requested = options->provider ? options->provider : "auto";
	const char *selected = requested;
	const char *names[] = { "cuda", "tensorrt", "coreml", "directml",
				"cpu" };
	OrtStatus *status;

	if (!strcmp(requested, "auto")) {
#ifdef __APPLE__
		selected = "coreml";
#elif defined(_WIN32) && defined(STRATA_DIRECTML)
		selected = "directml";
#else
		selected = "cuda";
#endif
		status = append_provider(ctx, selected, options->device_id);
		if (status) {
			fprintf(stderr,
				"strata: auto %s unavailable: %s; using cpu\n",
				selected, ctx->api->GetErrorMessage(status));
			ctx->api->ReleaseStatus(status);
			selected = "cpu";
		}
	} else if (ort_error(ctx,
			     append_provider(ctx, selected, options->device_id),
			     err)) {
		return -1;
	}
	for (size_t i = 0; i < sizeof(names) / sizeof(*names); ++i) {
		if (!strcmp(selected, names[i])) {
			ctx->provider = names[i];
			return 0;
		}
	}
	return strata_fail(err, "invalid provider");
}

static int set_fov_path(strata_context *ctx, const char *path,
			struct strata_error *err)
{
	size_t len = strlen(path);
	size_t stem = len >= 5 && !strcmp(path + len - 5, ".onnx") ? len - 5 :
								     len;

	ctx->model_path = malloc(len + 1);
	ctx->fov_path = malloc(stem + 10);
	if (!ctx->model_path || !ctx->fov_path)
		return strata_fail(err, "out of memory");
	memcpy(ctx->model_path, path, len + 1);
	memcpy(ctx->fov_path, path, stem);
	memcpy(ctx->fov_path + stem, ".fov.onnx", 10);
	return 0;
}

static int tensor_contract(strata_context *ctx, OrtSession *session, int input,
			   const char *expected_name, const int64_t *expected,
			   size_t rank, struct strata_error *err)
{
	OrtTypeInfo *type_info = NULL;
	const OrtTensorTypeAndShapeInfo *tensor_info = NULL;
	OrtAllocator *allocator = NULL;
	ONNXTensorElementDataType type;
	int64_t shape[4];
	size_t dimensions;
	char *name = NULL;
	OrtStatus *status;
	int ret = -1;

	ORT(GetAllocatorWithDefaultOptions(&allocator));
	status = input ? ctx->api->SessionGetInputName(session, 0, allocator,
						       &name) :
			 ctx->api->SessionGetOutputName(session, 0, allocator,
							&name);
	if (ort_error(ctx, status, err))
		goto out;
	status = input ? ctx->api->SessionGetInputTypeInfo(session, 0,
							   &type_info) :
			 ctx->api->SessionGetOutputTypeInfo(session, 0,
							    &type_info);
	if (ort_error(ctx, status, err))
		goto out;
	ORT(CastTypeInfoToTensorInfo(type_info, &tensor_info));
	if (!tensor_info)
		goto mismatch;
	ORT(GetTensorElementType(tensor_info, &type));
	ORT(GetDimensionsCount(tensor_info, &dimensions));
	if (strcmp(name, expected_name) ||
	    type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || dimensions != rank)
		goto mismatch;
	ORT(GetDimensions(tensor_info, shape, rank));
	for (size_t i = 0; i < rank; ++i) {
		if (expected[i] < 0 ? shape[i] >= 0 : shape[i] != expected[i])
			goto mismatch;
	}
	ret = 0;
	goto out;
mismatch:
	strata_fail(err, "ONNX tensor contract mismatch: %s", expected_name);
out:
	if (name)
		allocator->Free(allocator, name);
	if (type_info)
		ctx->api->ReleaseTypeInfo(type_info);
	return ret;
}

static int session_contract(strata_context *ctx, OrtSession *session, int fov,
			    struct strata_error *err)
{
	const int64_t image[] = { 1, 3, -1, -1 };
	const int64_t depth[] = { 1, 1, -1, -1 };
	const int64_t angles[] = { 1, 2 };
	size_t inputs, outputs;
	int ret = -1;

	ORT(SessionGetInputCount(session, &inputs));
	ORT(SessionGetOutputCount(session, &outputs));
	if (inputs != 1 || outputs != 1) {
		strata_fail(
			err,
			"ONNX tensor contract requires one input and one output");
		goto out;
	}
	if (tensor_contract(ctx, session, 1, STRATA_INPUT_NAME, image, 4, err))
		goto out;
	ret = tensor_contract(ctx, session, 0,
			      fov ? STRATA_FOV_OUTPUT_NAME : STRATA_OUTPUT_NAME,
			      fov ? angles : depth, fov ? 2u : 4u, err);
out:
	return ret;
}

static int load_session(strata_context *ctx, const char *path,
			OrtSession **session, int fov, struct strata_error *err)
{
	if (ort_error(ctx, create_session(ctx, path, session), err))
		return -1;
	if (session_contract(ctx, *session, fov, err)) {
		ctx->api->ReleaseSession(*session);
		*session = NULL;
		return -1;
	}
	return 0;
}

int strata_open(const struct strata_options *options, strata_context **out,
		struct strata_error *err)
{
	strata_context *ctx;
	int ret = -1;

	if (!options || !options->model_path || !out || options->device_id < 0)
		return strata_fail(err, "invalid session options");
	*out = NULL;
	if (strata_check_manifest(options->model_path, err))
		return -1;
	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return strata_fail(err, "out of memory");
	ctx->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (!ctx->api) {
		strata_fail(err, "ONNX Runtime API version mismatch");
		goto out;
	}
	ORT(CreateEnv(ORT_LOGGING_LEVEL_WARNING, "strata", &ctx->env));
	ORT(CreateSessionOptions(&ctx->options));
	ORT(SetIntraOpNumThreads(ctx->options, 4));
	ORT(SetSessionExecutionMode(ctx->options, ORT_SEQUENTIAL));
	ORT(SetSessionGraphOptimizationLevel(ctx->options, ORT_ENABLE_ALL));
	if (select_provider(ctx, options, err))
		goto out;
	ORT(CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
				&ctx->memory));
	if (load_session(ctx, options->model_path, &ctx->session, 0, err) ||
	    set_fov_path(ctx, options->model_path, err))
		goto out;
	*out = ctx;
	ret = 0;
out:
	if (ret)
		strata_close(ctx);
	return ret;
}

void strata_unload(strata_context *ctx)
{
	if (!ctx)
		return;
	if (ctx->fov)
		ctx->api->ReleaseSession(ctx->fov);
	if (ctx->session)
		ctx->api->ReleaseSession(ctx->session);
	ctx->fov = NULL;
	ctx->session = NULL;
}

void strata_close(strata_context *ctx)
{
	const OrtApi *api;

	if (!ctx)
		return;
	api = ctx->api;
	strata_unload(ctx);
	if (ctx->memory)
		api->ReleaseMemoryInfo(ctx->memory);
	if (ctx->options)
		api->ReleaseSessionOptions(ctx->options);
	if (ctx->env)
		api->ReleaseEnv(ctx->env);
	free(ctx->model_path);
	free(ctx->fov_path);
	free(ctx);
}

const char *strata_provider(const strata_context *ctx)
{
	return ctx ? ctx->provider : NULL;
}

static int run(strata_context *ctx, OrtSession *session,
	       const struct strata_tensor *input, const char *output_name,
	       OrtValue **output, struct strata_error *err)
{
	OrtValue *tensor = NULL;
	const char *name = STRATA_INPUT_NAME;
	int64_t shape[] = { 1, 3, input->height, input->width };
	size_t bytes = (size_t)3 * (size_t)input->height *
		       (size_t)input->width * sizeof(float);
	int ret = -1;

	ORT(CreateTensorWithDataAsOrtValue(
		ctx->memory, input->data, bytes, shape, 4,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &tensor));
	ORT(Run(session, NULL, &name, (const OrtValue *const *)&tensor, 1,
		&output_name, 1, output));
	ret = 0;
out:
	if (tensor)
		ctx->api->ReleaseValue(tensor);
	return ret;
}

static int estimate_focal(strata_context *ctx,
			  const struct strata_tensor *input, float *focal,
			  struct strata_error *err)
{
	struct strata_tensor crop = { 0 };
	OrtValue *output = NULL;
	float *angles;
	int ret = -1;

	if (!ctx->fov && load_session(ctx, ctx->fov_path, &ctx->fov, 1, err))
		goto out;
	if (strata_fov_input(input, &crop, err) ||
	    run(ctx, ctx->fov, &crop, STRATA_FOV_OUTPUT_NAME, &output, err))
		goto out;
	ORT(GetTensorMutableData(output, (void **)&angles));
	*focal = strata_focal(angles, input->width, input->height);
	ret = 0;
out:
	strata_tensor_free(&crop);
	if (output)
		ctx->api->ReleaseValue(output);
	return ret;
}

static int prediction(strata_context *ctx, OrtValue *output,
		      const struct strata_tensor *input, float **data,
		      struct strata_error *err)
{
	OrtTensorTypeAndShapeInfo *info = NULL;
	ONNXTensorElementDataType type;
	size_t rank;
	int64_t shape[4];
	int ret = -1;

	ORT(GetTensorTypeAndShape(output, &info));
	ORT(GetTensorElementType(info, &type));
	ORT(GetDimensionsCount(info, &rank));
	if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || rank != 4)
		goto mismatch;
	ORT(GetDimensions(info, shape, 4));
	if (shape[0] != 1 || shape[1] != 1 || shape[2] != input->height ||
	    shape[3] != input->width)
		goto mismatch;
	ORT(GetTensorMutableData(output, (void **)data));
	ret = 0;
	goto out;
mismatch:
	strata_fail(
		err,
		"model output must be float32 [1,1,padded_height,padded_width]");
out:
	if (info)
		ctx->api->ReleaseTensorTypeAndShapeInfo(info);
	return ret;
}

static int infer_frame(strata_context *ctx, const struct strata_image *image,
		       float focal, float exposure,
		       struct strata_result *result, struct strata_error *err)
{
	struct strata_tensor normalized = { 0 }, padded = { 0 };
	OrtValue *output = NULL;
	float *data;
	int ret = -1;

	if (strata_normalize(image, exposure, &normalized, err))
		goto out;
	if (focal == 0 && estimate_focal(ctx, &normalized, &focal, err))
		goto out;
	if (strata_pad(&normalized, &padded, err) ||
	    run(ctx, ctx->session, &padded, STRATA_OUTPUT_NAME, &output, err) ||
	    prediction(ctx, output, &padded, &data, err))
		goto out;
	ret = strata_postprocess(data, padded.width, image->width,
				 image->height, focal, result, err);
out:
	if (output)
		ctx->api->ReleaseValue(output);
	strata_tensor_free(&normalized);
	strata_tensor_free(&padded);
	return ret;
}

int strata_run(strata_context *ctx, const struct strata_frame *frames,
	       size_t frame_count, struct strata_result *results,
	       struct strata_error *err)
{
	const struct strata_frame *frame = frames;

	if (frame_count != 1)
		return strata_fail(
			err,
			"P0 supports one frame; temporal windows require a temporal model contract");
	if (!ctx || !frame || !frame->image || !frame->image->rgba ||
	    !results || !isfinite(frame->camera.fx_px) ||
	    frame->camera.fx_px < 0 || !isfinite(frame->camera.fy_px) ||
	    frame->camera.fy_px < 0 || !isfinite(frame->camera.cx_px) ||
	    !isfinite(frame->camera.cy_px) ||
	    (frame->camera.fx_px > 0 && frame->camera.fy_px <= 0) ||
	    (frame->camera.fx_px == 0 &&
	     (frame->camera.fy_px != 0 || frame->camera.cx_px != 0 ||
	      frame->camera.cy_px != 0)) ||
	    !isfinite(frame->exposure_stops) ||
	    fabsf(frame->exposure_stops) > 100 ||
	    !isfinite(frame->camera.sensor_width_mm) ||
	    frame->camera.sensor_width_mm < 0 ||
	    !isfinite(frame->camera.sensor_height_mm) ||
	    frame->camera.sensor_height_mm < 0 ||
	    !isfinite(frame->timestamp_seconds))
		return strata_fail(err, "invalid frame arguments");
	memset(results, 0, sizeof(*results));
	if (!ctx->session &&
	    (strata_check_manifest(ctx->model_path, err) ||
	     load_session(ctx, ctx->model_path, &ctx->session, 0, err)))
		return -1;
	if (infer_frame(ctx, frame->image, frame->camera.fx_px,
			frame->exposure_stops, results, err))
		return -1;
	results->frame.index = frame->index;
	results->frame.timestamp_seconds = frame->timestamp_seconds;
	results->frame.linear = frame->image->linear;
	if (frame->camera.fx_px > 0)
		results->frame.camera = frame->camera;
	results->frame.camera.sensor_width_mm = frame->camera.sensor_width_mm;
	results->frame.camera.sensor_height_mm = frame->camera.sensor_height_mm;
	results->frame.camera.focal_source = frame->camera.fx_px == 0 ?
						     STRATA_FOCAL_ESTIMATED :
						     STRATA_FOCAL_PROVIDED;
	return 0;
}
