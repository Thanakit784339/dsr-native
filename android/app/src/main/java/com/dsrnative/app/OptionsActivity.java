package com.dsrnative.app;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.view.Gravity;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The real launcher activity: a plain options screen (programmatic views, no layout resources --
 * there's little enough here that a layout file would just be more to maintain) shown before the
 * game itself starts. "Start Game" upserts the chosen values into settings.txt -- existing lines
 * and comments for keys not shown here are left untouched -- then launches GameActivity.
 */
public class OptionsActivity extends Activity {
    private Spinner mFps, mPresentMode, mFaceButtons, mMipSkip, mMaxAnisotropy;
    private CheckBox mAsyncPipelines, mAutoCapture, mProfile, mTouchControls, mDocked, mFpsCounter;
    private CheckBox mAudioSmooth, mFastCores, mRecordThread, mPinRenderThread, mAstcTranscode, mMaliBarriers;
    private FloatSlider mGamma, mExposure, mRenderScale;

    private File settingsFile() { return AppFiles.resolve(this, "settings.txt"); }

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        if (!AppFiles.hasStorageAccess(this)) requestStorageAccessBestEffort();  // GameActivity asks again if refused

        final int pad = dp(16);
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);
        scroll.addView(root);

        root.addView(title("DSR Native — Settings"));

        mFps = labeledSpinner(root, "Frame rate cap", new String[] {"30", "45", "60"});
        tip(root, "30 matches the original Switch rate and is easiest to hold steady. 45/60 move smoother "
                + "but need more GPU headroom to avoid dropping frames in heavy areas.");

        mAsyncPipelines = checkBox(root, "Async pipeline compilation");
        tip(root, "On: a newly-seen shader compiles in the background, no hitch (it may draw a frame late). "
                + "Off: compiles immediately on the spot, which can stutter the first time something new appears.");

        mAutoCapture = checkBox(root, "Auto frame capture (diagnostics)");
        tip(root, "On: periodically saves frame captures for debugging. Off: only the manual capture "
                + "(press both sticks in, L3 + R3) still works. No effect on performance either way.");

        mProfile = checkBox(root, "CPU profiling log (diagnostics)");
        tip(root, "On: logs where CPU time goes every 10 s, costs about 1% performance. Off: no profiling log.");

        mGamma = labeledSlider(root, "Gamma: ", 0.5f, 3.0f, 0.05f, 1.00f);
        tip(root, "Extra brightness curve on top of the game's own. 1.0 = unchanged. Lower darkens, higher "
                + "(1.2-1.6) lifts shadow detail. Pure display tweak, no performance effect.");

        mExposure = labeledSlider(root, "Exposure: ", 0.25f, 4.0f, 0.05f, 1.37f);
        tip(root, "Scales scene brightness before tone mapping. Lower = darker overall image, higher = "
                + "brighter. Pure display tweak, no performance effect.");

        mMaxAnisotropy = labeledSpinner(root, "Max anisotropic filtering", new String[] {"1", "2", "4", "8", "16"});
        tip(root, "Caps texture sharpness at a grazing angle (floors, ground stretching into the distance). "
                + "Lower = blurrier at an angle but cheaper for the GPU. Higher = sharper but more GPU sampling "
                + "cost. 4 is a good middle ground; 1 disables it entirely.");

        mPresentMode = labeledSpinner(root, "Present mode",
                new String[] {"mailbox", "fifo", "fifo_relaxed", "immediate"});
        tip(root, "mailbox: low latency, no tearing (recommended). fifo: traditional vsync, can add input lag. "
                + "fifo_relaxed: vsync but allows tearing if a frame runs late. immediate: no vsync, lowest "
                + "latency, can tear.");

        mFaceButtons = labeledSpinner(root, "Face buttons", new String[] {"label", "position"});
        tip(root, "label: buttons shown/mapped by their printed letter (A/B/X/Y), Xbox-style. position: mapped "
                + "by physical position instead, matching a Nintendo-layout controller.");

        mTouchControls = checkBox(root, "Show on-screen touch controls");
        tip(root, "On: on-screen buttons shown (a toggle button, top-right, can hide/show them in-game too). "
                + "Off: fully hidden, use a physical controller.");

        mFpsCounter = checkBox(root, "Show FPS counter");
        tip(root, "On: small FPS counter overlay, top-left. Off: no overlay. No real performance difference.");

        mDocked = checkBox(root, "Docked mode");
        tip(root, "On: the real Switch flag telling the game it's docked -- most games including this one "
                + "render at a higher internal resolution, more demanding on the GPU. Off: handheld 720p, "
                + "lighter load.");

        mRenderScale = labeledSlider(root, "Render scale: ", 0.1f, 1.0f, 0.05f, 1.00f);
        tip(root, "Fraction of the game's normal resolution actually rendered, then upscaled to fill the "
                + "screen (experimental). Lower = fewer pixels to draw, more fps, softer/blockier image. "
                + "1.0 = full resolution, no upscale.");

        mMipSkip = labeledSpinner(root, "Texture mip levels to skip", new String[] {"0", "1", "2", "3", "4"});
        tip(root, "Skips this many of a big texture's own top (highest-detail) mip levels so they're never "
                + "even loaded. 0 = full quality. Higher = softer textures up close but a lot less data to "
                + "stream during loading/streaming spikes.");

        mAudioSmooth = checkBox(root, "Smooth audio buffer pacing");
        tip(root, "Experimental. On: releases audio buffers to the game at a smoothed pace. Off: default "
                + "timing (recommended unless you hear audio stutter).");

        mFastCores = checkBox(root, "Keep frame-critical threads off efficiency cores");
        tip(root, "On (recommended): the game's main/recording/submit threads stay on the fast performance "
                + "cores. Off: Android schedules them freely, which can land them on slow efficiency cores.");

        mRecordThread = checkBox(root, "Record GPU commands on a second thread");
        tip(root, "On (recommended): GPU command recording happens off the main thread, usually faster. Off: "
                + "everything on one thread -- slower but can be useful for isolating a bug.");

        mPinRenderThread = checkBox(root, "Pin render thread to the fastest CPU core(s)");
        tip(root, "Chip-dependent -- try both. On: forces the render thread onto the fastest cores. Off: lets "
                + "Android choose (was faster on some chips, slower on others in testing).");

        mMaliBarriers = checkBox(root, "Narrow GPU sync barriers for tile-based GPUs (Mali)");
        tip(root, "On (recommended): lets the GPU overlap more work between render passes on tile-based chips "
                + "like Mali -- real fps win, no known downside. Off: the old, maximally-conservative "
                + "synchronization (try this only if you suspect a rendering glitch is related to it).");

        mAstcTranscode = checkBox(root, "ASTC texture transcode (EXPERIMENTAL)");
        tip(root, "On: re-compresses textures to ASTC instead of the plain RGBA8 fallback, saving real GPU "
                + "bandwidth. Known to cause missing textures and text on some GPU drivers -- confirmed broken "
                + "on at least one device/driver combo. Off (recommended, default): always use the "
                + "proven-stable RGBA8 fallback.");

        loadCurrentValues();

        Button start = new Button(this);
        start.setText("Start Game");
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(24);
        start.setLayoutParams(lp);
        start.setOnClickListener(v -> { saveAndLaunch(); });
        root.addView(start);

        setContentView(scroll);
    }

    private void requestStorageAccessBestEffort() {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R)
                startActivity(new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        Uri.parse("package:" + getPackageName())));
            else
                requestPermissions(new String[] {Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        } catch (Exception ignored) {
            // No handler for the settings screen, or permission denied outright: settings.txt falls back
            // to the app-private folder (settingsFile() above), same as the native side already does.
        }
    }

    // ---- reading the existing file (if any) so the screen reflects what's actually set ----------------
    private void loadCurrentValues() {
        setSpinner(mFps, "30");
        mAsyncPipelines.setChecked(true);
        mAutoCapture.setChecked(false);
        mProfile.setChecked(false);
        mGamma.set(1.00f);
        mExposure.set(1.37f);
        setSpinner(mMaxAnisotropy, "4");
        setSpinner(mPresentMode, "mailbox");
        setSpinner(mFaceButtons, "label");
        mTouchControls.setChecked(true);
        mDocked.setChecked(false);
        mRenderScale.set(1.00f);
        setSpinner(mMipSkip, "1");
        mFpsCounter.setChecked(true);
        mAudioSmooth.setChecked(false);
        mFastCores.setChecked(true);
        mRecordThread.setChecked(true);
        mPinRenderThread.setChecked(false);
        mAstcTranscode.setChecked(false);
        mMaliBarriers.setChecked(true);

        File f = settingsFile();
        if (f.exists()) {
            try (BufferedReader r = new BufferedReader(new FileReader(f))) {
                String line;
                while ((line = r.readLine()) != null) {
                    Matcher m = KV.matcher(line);
                    if (!m.matches()) continue;
                    final String key = m.group(1), value = m.group(2);
                    switch (key) {
                        case "fps": setSpinner(mFps, value); break;
                        case "async_pipelines": mAsyncPipelines.setChecked(!"0".equals(value)); break;
                        case "auto_capture": mAutoCapture.setChecked(!"0".equals(value)); break;
                        case "profile": mProfile.setChecked(!"0".equals(value)); break;
                        case "gamma": mGamma.setFromText(value); break;
                        case "exposure": mExposure.setFromText(value); break;
                        case "max_anisotropy": setSpinner(mMaxAnisotropy, roundedIntString(value)); break;
                        case "present_mode": setSpinner(mPresentMode, value); break;
                        case "face_buttons": setSpinner(mFaceButtons, value); break;
                        case "touch_controls": mTouchControls.setChecked(!"0".equals(value)); break;
                        case "docked": mDocked.setChecked(!"0".equals(value)); break;
                        case "render_scale": mRenderScale.setFromText(value); break;
                        case "texture_mip_skip": setSpinner(mMipSkip, value); break;
                        case "fps_counter": mFpsCounter.setChecked(!"0".equals(value)); break;
                        case "audio_smooth": mAudioSmooth.setChecked(!"0".equals(value)); break;
                        case "fast_cores": mFastCores.setChecked(!"0".equals(value)); break;
                        case "record_thread": mRecordThread.setChecked(!"0".equals(value)); break;
                        case "pin_render_thread": mPinRenderThread.setChecked(!"0".equals(value)); break;
                        case "astc_transcode": mAstcTranscode.setChecked(!"0".equals(value)); break;
                        case "mali_barriers": mMaliBarriers.setChecked(!"0".equals(value)); break;
                        default: break;
                    }
                }
            } catch (IOException ignored) {
            }
        }
    }

    private static String roundedIntString(String floatText) {
        try {
            return String.valueOf(Math.round(Float.parseFloat(floatText)));
        } catch (NumberFormatException e) {
            return floatText;
        }
    }

    // ---- writing back: only the keys this screen has a widget for are touched --------------------------
    private static final Pattern KV = Pattern.compile("^\\s*([a-z_0-9]+)\\s*=\\s*(\\S+)\\s*$");

    private void saveAndLaunch() {
        List<String> lines = new ArrayList<>();
        File f = settingsFile();
        if (f.exists()) {
            try (BufferedReader r = new BufferedReader(new FileReader(f))) {
                String line;
                while ((line = r.readLine()) != null) lines.add(line);
            } catch (IOException ignored) {
            }
        }
        upsert(lines, "fps", selected(mFps));
        upsert(lines, "async_pipelines", mAsyncPipelines.isChecked() ? "1" : "0");
        upsert(lines, "auto_capture", mAutoCapture.isChecked() ? "1" : "0");
        upsert(lines, "profile", mProfile.isChecked() ? "1" : "0");
        upsert(lines, "gamma", mGamma.text());
        upsert(lines, "exposure", mExposure.text());
        upsert(lines, "max_anisotropy", selected(mMaxAnisotropy));
        upsert(lines, "present_mode", selected(mPresentMode));
        upsert(lines, "face_buttons", selected(mFaceButtons));
        upsert(lines, "touch_controls", mTouchControls.isChecked() ? "1" : "0");
        upsert(lines, "docked", mDocked.isChecked() ? "1" : "0");
        upsert(lines, "render_scale", mRenderScale.text());
        upsert(lines, "texture_mip_skip", selected(mMipSkip));
        upsert(lines, "fps_counter", mFpsCounter.isChecked() ? "1" : "0");
        upsert(lines, "audio_smooth", mAudioSmooth.isChecked() ? "1" : "0");
        upsert(lines, "fast_cores", mFastCores.isChecked() ? "1" : "0");
        upsert(lines, "record_thread", mRecordThread.isChecked() ? "1" : "0");
        upsert(lines, "pin_render_thread", mPinRenderThread.isChecked() ? "1" : "0");
        upsert(lines, "astc_transcode", mAstcTranscode.isChecked() ? "1" : "0");
        upsert(lines, "mali_barriers", mMaliBarriers.isChecked() ? "1" : "0");

        File parent = f.getParentFile();
        if (parent != null) parent.mkdirs();
        try (FileWriter w = new FileWriter(f)) {
            for (String line : lines) w.write(line + "\n");
        } catch (IOException e) {
            // Can't persist settings, but no reason to block play over it -- the native side falls back
            // to its own defaults (and the app-private folder) exactly as if this file didn't exist.
        }

        startActivity(new Intent(this, GameActivity.class));
        finish();
    }

    private static void upsert(List<String> lines, String key, String value) {
        Pattern p = Pattern.compile("^\\s*" + key + "\\s*=.*$");
        for (int i = 0; i < lines.size(); i++) {
            if (p.matcher(lines.get(i)).matches()) {
                lines.set(i, key + " = " + value);
                return;
            }
        }
        lines.add(key + " = " + value);
    }

    // ---- small view-building helpers (programmatic UI: this screen is simple enough that a layout
    // resource would just be one more file to keep in sync) -----------------------------------------
    private TextView title(String text) {
        TextView t = new TextView(this);
        t.setText(text);
        t.setTextSize(20);
        t.setPadding(0, 0, 0, dp(16));
        return t;
    }

    private Spinner labeledSpinner(LinearLayout root, String label, String[] values) {
        root.addView(label(label));
        Spinner s = new Spinner(this);
        s.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, values));
        root.addView(s);
        return s;
    }

    private CheckBox checkBox(LinearLayout root, String label) {
        CheckBox cb = new CheckBox(this);
        cb.setText(label);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(12);
        cb.setLayoutParams(lp);
        root.addView(cb);
        return cb;
    }

    /** A small dimmed explanatory line under a control -- what lower/higher or on/off actually does. */
    private void tip(LinearLayout root, String text) {
        TextView t = new TextView(this);
        t.setText(text);
        t.setTextSize(12);
        t.setTextColor(0xFFA0A0A0);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.bottomMargin = dp(4);
        t.setLayoutParams(lp);
        root.addView(t);
    }

    /** A float value picked via a SeekBar (integer progress) instead of hand-typed text. */
    private static class FloatSlider {
        SeekBar bar;
        TextView valueText;
        String prefix;
        float min, step;

        float get() { return min + bar.getProgress() * step; }

        void set(float v) {
            bar.setProgress(Math.round((v - min) / step));
            updateText();
        }

        void setFromText(String text) {
            try {
                set(Float.parseFloat(text));
            } catch (NumberFormatException ignored) {
            }
        }

        String text() { return String.format(Locale.US, "%.2f", get()); }

        void updateText() { valueText.setText(prefix + String.format(Locale.US, "%.2f", get())); }
    }

    private FloatSlider labeledSlider(LinearLayout root, String prefix, float min, float max, float step, float def) {
        FloatSlider fs = new FloatSlider();
        fs.min = min;
        fs.step = step;
        fs.prefix = prefix;

        fs.valueText = new TextView(this);
        fs.valueText.setTextSize(15);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(14);
        fs.valueText.setLayoutParams(lp);
        root.addView(fs.valueText);

        SeekBar sb = new SeekBar(this);
        sb.setMax(Math.round((max - min) / step));
        sb.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) { fs.updateText(); }
            @Override public void onStartTrackingTouch(SeekBar seekBar) { }
            @Override public void onStopTrackingTouch(SeekBar seekBar) { }
        });
        root.addView(sb);
        fs.bar = sb;
        fs.set(def);
        return fs;
    }

    private TextView label(String text) {
        TextView t = new TextView(this);
        t.setText(text);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(14);
        t.setLayoutParams(lp);
        t.setGravity(Gravity.START);
        return t;
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }

    private static void setSpinner(Spinner s, String value) {
        ArrayAdapter adapter = (ArrayAdapter) s.getAdapter();
        if (adapter == null) return;
        int pos = adapter.getPosition(value);
        if (pos >= 0) s.setSelection(pos);
    }

    private static String selected(Spinner s) {
        Object sel = s.getSelectedItem();
        return sel != null ? sel.toString() : "";
    }
}
