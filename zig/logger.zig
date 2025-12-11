const std = @import("std");

/// Logging levels for build output
pub const LogLevel = enum(u8) {
    silent = 0, // No output
    err = 1, // Only errors
    warn = 2, // Errors and warnings
    info = 3, // Errors, warnings, and info
    debug = 4, // All output including debug
    trace = 5, // Most verbose output

    pub fn fromString(s: []const u8) ?LogLevel {
        if (std.mem.eql(u8, s, "silent")) return .silent;
        if (std.mem.eql(u8, s, "error")) return .err;
        if (std.mem.eql(u8, s, "warn")) return .warn;
        if (std.mem.eql(u8, s, "info")) return .info;
        if (std.mem.eql(u8, s, "debug")) return .debug;
        if (std.mem.eql(u8, s, "trace")) return .trace;
        return null;
    }

    pub fn toString(self: LogLevel) []const u8 {
        return switch (self) {
            .silent => "silent",
            .err => "error",
            .warn => "warn",
            .info => "info",
            .debug => "debug",
            .trace => "trace",
        };
    }
};

/// Global logger instance
var global_log_level: LogLevel = .info;

/// Logger functions
pub const Log = struct {
    pub fn setLevel(level: LogLevel) void {
        global_log_level = level;
    }

    pub fn err(comptime fmt: []const u8, args: anytype) void {
        if (@intFromEnum(global_log_level) >= @intFromEnum(LogLevel.err)) {
            std.debug.print("[ERROR] " ++ fmt ++ "\n", args);
        }
    }

    pub fn warn(comptime fmt: []const u8, args: anytype) void {
        if (@intFromEnum(global_log_level) >= @intFromEnum(LogLevel.warn)) {
            std.debug.print("[WARN] " ++ fmt ++ "\n", args);
        }
    }

    pub fn info(comptime fmt: []const u8, args: anytype) void {
        if (@intFromEnum(global_log_level) >= @intFromEnum(LogLevel.info)) {
            std.debug.print("[INFO] " ++ fmt ++ "\n", args);
        }
    }

    pub fn debug(comptime fmt: []const u8, args: anytype) void {
        if (@intFromEnum(global_log_level) >= @intFromEnum(LogLevel.debug)) {
            std.debug.print("[DEBUG] " ++ fmt ++ "\n", args);
        }
    }

    pub fn trace(comptime fmt: []const u8, args: anytype) void {
        if (@intFromEnum(global_log_level) >= @intFromEnum(LogLevel.trace)) {
            std.debug.print("[TRACE] " ++ fmt ++ "\n", args);
        }
    }
};
