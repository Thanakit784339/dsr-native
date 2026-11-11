package com.dsrnative.app;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Environment;

import java.io.File;

/** Where this app's user-editable files live: Download/DSRNative if storage access was granted, the
 * app-private folder otherwise -- same fallback the native side (android_main.cpp) already uses. */
final class AppFiles {
    private static final String SHARED_DIR = "/storage/emulated/0/Download/DSRNative";

    private AppFiles() {}

    static boolean hasStorageAccess(Context c) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) return Environment.isExternalStorageManager();
        return c.checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    static File resolve(Context c, String name) {
        File shared = new File(SHARED_DIR, name);
        if (hasStorageAccess(c) || (shared.getParentFile() != null && shared.getParentFile().exists()))
            return shared;
        return new File(c.getExternalFilesDir(null), name);
    }
}
