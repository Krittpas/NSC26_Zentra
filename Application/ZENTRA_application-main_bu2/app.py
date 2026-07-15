import os
import sys
import threading
import time
import webview
from pathlib import Path
    
# The ZENTRA AI modules print emoji / Thai (✅ ⚠️ 🪖 →) at import and at
# runtime. On a Windows cp1252 console these prints raise
# UnicodeEncodeError, which would crash pipeline startup. Force UTF-8 with
# errors='replace' so logging can never take the app down.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def start_server():
    import uvicorn
    from server.api import app as fastapi_app
    uvicorn.run(
        fastapi_app,
        host="127.0.0.1", #192.168.1.64
        port=7788,
        log_level="warning",
        # Suppress uvicorn shutdown errors on Windows
        timeout_graceful_shutdown=2,
    )


class JsApi:
    def open_file_dialog(self):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="เลือกไฟล์วิดีโอ",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path or ""

    def save_file(self, content: str, filename: str):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title="บันทึกไฟล์",
            defaultextension=".csv",
            initialfile=filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            Path(path).write_text(content, encoding="utf-8-sig")
        return path or ""

    def save_binary(self, b64_content: str, filename: str):
        """Open a native Save dialog and write base64-decoded bytes (PDF, etc.).
        WebView2 does not reliably trigger blob downloads, so binary files are
        saved through this bridge instead."""
        import base64
        import tkinter as tk
        from tkinter import filedialog
        ext = ("." + filename.rsplit(".", 1)[-1]) if "." in filename else ""
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title="บันทึกไฟล์",
            defaultextension=ext,
            initialfile=filename,
            filetypes=[("PDF", "*.pdf"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            try:
                Path(path).write_bytes(base64.b64decode(b64_content))
            except Exception as e:
                print(f"[JsApi] save_binary error: {e}")
                return ""
        return path or ""

    def toggle_fullscreen(self):
        if webview.windows:
            webview.windows[0].toggle_fullscreen()

    def close_app(self):
        """Quit from an in-app button. destroy() fires the window 'closing'
        event, which is wired to shutdown_pipeline() — so the camera/pipeline
        and LINE sender are released cleanly, same as clicking the native X."""
        if webview.windows:
            webview.windows[0].destroy()


def shutdown_pipeline():
    """Stop the AI pipeline + background threads cleanly on window close.

    The uvicorn server runs in a daemon thread, so its FastAPI shutdown
    event is not guaranteed to fire when the main thread exits. We stop
    the pipeline explicitly here to release the camera and flush LINE
    alerts before the process ends.
    """
    try:
        import server.api as api
        if getattr(api, "_broadcaster", None):
            api._broadcaster.stop()
        if getattr(api, "pipeline", None):
            api.pipeline.stop()
        try:
            from alerts.line_notify import stop_sender
            stop_sender()
        except Exception:
            pass
        print("[App] Clean shutdown complete")
    except Exception as e:
        print(f"[App] shutdown warning: {e}")


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2.0)

    # If the server requires a token (ZENTRA_API_TOKEN), hand it to the UI on the
    # initial URL so the same desktop window authenticates automatically. The UI
    # persists it and attaches it to every fetch/WebSocket call. No token → plain URL.
    _token = os.getenv("ZENTRA_API_TOKEN", "").strip()
    _url = "http://127.0.0.1:7788/" + (f"?token={_token}" if _token else "")

    window = webview.create_window(
        title="ZENTRA Safety AI System",
        url=_url,
        # Open full-screen (kiosk-style) instead of a floating window. In
        # full-screen there is no native title bar / X, so the in-app quit button
        # (top-right) is the way out; F11 / toggle_fullscreen() also exits.
        # width/height are the size the window falls back to if full-screen is left.
        fullscreen=True,
        width=1280,
        height=800,
        min_size=(1024, 640),
        js_api=JsApi(),
        background_color="#0d1b2a",
    )
    # Stop the pipeline as soon as the window begins closing
    window.events.closing += shutdown_pipeline

    webview.start(debug=False)

    # Safety net: also stop after the GUI loop returns
    shutdown_pipeline()
