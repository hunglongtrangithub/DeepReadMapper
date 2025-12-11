const std = @import("std");
const logger = @import("zig/logger.zig");
const builder_mod = @import("zig/builder.zig");

const Log = logger.Log;
const LogLevel = logger.LogLevel;
const Builder = builder_mod.Builder;

/// Expected conda environment name that should be activated
const CONDA_ENV = "DeepReadMapper";
/// Highest-level build output directory
const ZIG_OUT = "zig-out";
/// Output directory for binaries
const BIN_OUT = ZIG_OUT ++ "/bin";
/// Output directory for object files and dependency files
const OBJ_OUT = ZIG_OUT ++ "/obj";

pub fn build(b: *std.Build) !void {
    // Configure logging level from command line
    const log_level_str = b.option([]const u8, "log", "Set log level (silent, error, warn, info, debug, trace)") orelse "info";
    const log_level = LogLevel.fromString(log_level_str) orelse {
        std.debug.print("Invalid log level: {s}. Valid levels: silent, error, warn, info, debug, trace\n", .{log_level_str});
        std.process.exit(1);
    };
    Log.setLevel(log_level);

    const conda_prefix = std.posix.getenv("CONDA_PREFIX") orelse {
        Log.err("CONDA_PREFIX not set. Please activate conda environment: {s}", .{CONDA_ENV});
        std.process.exit(1);
    };

    if (!std.mem.eql(u8, std.fs.path.basename(conda_prefix), CONDA_ENV)) {
        Log.err("Incorrect conda environment: {s}. Please activate conda environment: {s}", .{ conda_prefix, CONDA_ENV });
        std.process.exit(1);
    }

    Log.info("Using conda prefix: {s}", .{conda_prefix});

    const gxx_path = b.findProgram(&[_][]const u8{"g++"}, &[_][]const u8{
        b.fmt("{s}/bin", .{conda_prefix}),
        "/usr/bin",
    }) catch "g++";

    Log.info("Using g++ path: {s}", .{gxx_path});

    // Initialize Builder
    const builder = Builder.init(b, gxx_path, conda_prefix, BIN_OUT, OBJ_OUT);

    const common_sources = [_][]const u8{
        "external/cnpy/cnpy.cpp",
        "src/utils/parse_inputs.cpp",
        "src/utils/utils.cpp",
        "src/utils/tok2index.cpp",
        "src/utils/post_processor.cpp",
        "src/utils/metrics.cpp",
        "src/utils/reranker.cpp",
        "src/inference/vectorize.cpp",
        "src/inference/fast_model.cpp",
        "src/inference/preprocess.cpp",
    };

    const common_flags = [_][]const u8{
        // "-ftime-report",
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-march=native",
        "-Wall",
        "-lstdc++", // for C++ standard library
        "-lz", // zlib for cnpy
    };

    const common_includes = [_][]const u8{
        "-Iincludes",
        "-Iincludes/inference",
        "-Iincludes/utils",
        "-Iincludes/hnswlib_dir",
        "-Iincludes/gann_hnsw",
        "-Iincludes/hnswpq",
        "-Iexternal/cereal/include",
        "-Iexternal/cnpy",
    };

    // Compile common sources first to be used by multiple executables
    var all_object_files = std.array_list.Managed([]const u8).init(b.allocator);
    var all_object_steps = std.array_list.Managed(*std.Build.Step.Run).init(b.allocator);

    for (common_sources) |source| {
        const obj_path = b.fmt("{s}/{s}.o", .{ OBJ_OUT, builder.sourceToObjectName(source) });
        try all_object_files.append(obj_path);

        if (try builder.createConditionalObjectCmd(source, &common_flags, &common_includes)) |obj_cmd| {
            obj_cmd.step.dependOn(&builder.mkdir_obj_cmd.step);
            try all_object_steps.append(obj_cmd);
        }
    }

    // Build executables using the Builder
    const pipeline_cmd = try builder.build(
        &[_][]const u8{ "src/main.cpp", "src/hnswpq/search.cpp" },
        all_object_files.items,
        all_object_steps.items,
        "pipeline",
        &[_][]const u8{ "-lopenvino", "-lomp", "-lfaiss" },
        &common_flags,
        &common_includes,
    );

    const inference_cmd = try builder.build(
        &[_][]const u8{"src/inference/test_inference.cpp"},
        all_object_files.items,
        all_object_steps.items,
        "inference",
        &[_][]const u8{"-lopenvino"},
        &common_flags,
        &common_includes,
    );

    const hnswpq_cmd = try builder.build(
        &[_][]const u8{"src/hnswpq/index.cpp"},
        all_object_files.items,
        all_object_steps.items,
        "hnswpq_index",
        &[_][]const u8{ "-lopenvino", "-lomp", "-lfaiss" },
        &common_flags,
        &common_includes,
    );

    // Set up build step dependencies
    const install_step = b.getInstallStep();
    install_step.dependOn(&pipeline_cmd.step);
    install_step.dependOn(&inference_cmd.step);
    install_step.dependOn(&hnswpq_cmd.step);

    // Individual build steps
    const pipeline_step = b.step("pipeline", "Build only the pipeline executable");
    pipeline_step.dependOn(&pipeline_cmd.step);

    const inference_step = b.step("inference", "Build only the inference executable");
    inference_step.dependOn(&inference_cmd.step);

    const index_step = b.step("index", "Build only the index executable");
    index_step.dependOn(&hnswpq_cmd.step);

    // Build step for common object files
    const objects_step = b.step("objects", "Build common object files");
    for (all_object_steps.items) |obj_step| {
        objects_step.dependOn(&obj_step.step);
    }

    // Clean step to remove build artifacts
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&[_][]const u8{ "rm", "-rf", BIN_OUT, OBJ_OUT });
    clean_step.dependOn(&clean_cmd.step);
}
