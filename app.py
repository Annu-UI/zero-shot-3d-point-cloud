import gradio as gr
import os
import time
import cv2
import numpy as np
from src.pipeline import PointCloudPipeline

print("Loading pipeline... this may take a minute on first run")
pipeline = PointCloudPipeline()
print("Pipeline ready!")

def process_image(image, object_classes_text, progress=gr.Progress()):
    if image is None:
        return "<p style='color:#ff4d4d;font-family:monospace;padding:20px'>⚠ Please upload an image</p>", ""
    
    if not object_classes_text.strip():
        return "<p style='color:#ff4d4d;font-family:monospace;padding:20px'>⚠ Please enter object classes</p>", ""
    
    classes = [c.strip() for c in object_classes_text.split(",") if c.strip()]
    
    progress(0.1, desc="Saving image...")
    temp_path = "data/images/temp_input.jpg"
    os.makedirs("data/images", exist_ok=True)
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    progress(0.2, desc="Detecting objects...")
    
    start = time.time()
    results = pipeline.run(temp_path, classes)
    elapsed = time.time() - start
    
    progress(0.9, desc="Rendering 3D point cloud...")
    
    if results is None:
        return (
            f"<p style='color:#ff4d4d;font-family:monospace;padding:20px'>"
            f"⚠ No objects found for: {classes}<br>Try different class names or a clearer image</p>",
            ""
        )
    
    # Read and inject HTML inline so it renders directly in Gradio
    with open(results["html_path"], 'r') as f:
        raw_html = f.read()

    # Wrap in iframe trick — embed the plotly HTML directly
    import base64
    encoded = base64.b64encode(raw_html.encode()).decode()
    iframe_html = f'''
        <iframe 
            src="data:text/html;base64,{encoded}" 
            width="100%" 
            height="600px" 
            style="border:none; border-radius:12px; background:#0d0d0d;"
        ></iframe>
    '''

    m = results["metrics"]
    metrics_html = f"""
<div style="
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    background: #0d0d0d;
    color: #e0e0e0;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #2a2a2a;
    line-height: 2;
    font-size: 13px;
">
    <div style="color:#00ff88;font-size:15px;margin-bottom:12px;letter-spacing:2px">
        ◆ POINT CLOUD QUALITY REPORT
    </div>
    <div style="color:#888;margin-bottom:16px">
        Objects detected: <span style="color:#fff">{', '.join(results['labels_detected'])}</span>
        &nbsp;|&nbsp; Total time: <span style="color:#fff">{elapsed:.2f}s</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 32px">
        <div>
            <span style="color:#666">Mask coverage</span><br>
            <span style="color:#00ff88;font-size:22px;font-weight:bold">{m['mask_coverage_pct']}%</span>
        </div>
        <div>
            <span style="color:#666">Quality score</span><br>
            <span style="color:#00ff88;font-size:22px;font-weight:bold">{m['quality_score']}/100</span>
        </div>
        <div>
            <span style="color:#666">3D points</span><br>
            <span style="color:#fff;font-size:18px">{m['points_after_denoise']:,}</span>
        </div>
        <div>
            <span style="color:#666">Point density</span><br>
            <span style="color:#fff;font-size:18px">{m['point_density_per_px2']} pts/px²</span>
        </div>
        <div>
            <span style="color:#666">Noise removed</span><br>
            <span style="color:#fff;font-size:18px">{m['noise_ratio_pct']}%</span>
        </div>
        <div>
            <span style="color:#666">Depth variance</span><br>
            <span style="color:#fff;font-size:18px">{m['depth_variance']}</span>
        </div>
    </div>
    <div style="margin-top:16px;padding-top:16px;border-top:1px solid #2a2a2a;color:#555;font-size:11px">
        Detection: {m['detection_time']}s &nbsp;|&nbsp; 
        Segmentation: {m['segmentation_time']}s &nbsp;|&nbsp; 
        Depth: {m['depth_time']}s &nbsp;|&nbsp; 
        Projection: {m['projection_time']}s
    </div>
</div>
"""
    progress(1.0, desc="Done!")
    return iframe_html, metrics_html


# --- Custom CSS ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');

body, .gradio-container {
    background: #080808 !important;
    font-family: 'Inter', sans-serif !important;
}

.main-header {
    text-align: center;
    padding: 40px 20px 20px;
}

.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
}

.main-header p {
    color: #555;
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.tag {
    display: inline-block;
    background: #111;
    border: 1px solid #222;
    color: #00ff88;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 0 4px;
    letter-spacing: 1px;
}

button.primary-btn {
    background: #00ff88 !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    height: 48px !important;
    transition: all 0.2s ease !important;
}

button.primary-btn:hover {
    background: #00cc6a !important;
    transform: translateY(-1px) !important;
}

.gr-box, .gr-panel {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 12px !important;
}

label {
    color: #888 !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-family: 'Space Mono', monospace !important;
}

input, textarea {
    background: #111 !important;
    border: 1px solid #222 !important;
    color: #fff !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}
"""

# --- Build UI ---
with gr.Blocks(css=custom_css, title="3D Point Cloud Generator") as demo:

    gr.HTML("""
    <div class="main-header">
        <h1>ZERO-SHOT 3D POINT CLOUD</h1>
        <p>Single image → interactive 3D reconstruction</p>
        <div style="margin-top:12px">
            <span class="tag">YOLO-WORLD</span>
            <span class="tag">SAM</span>
            <span class="tag">DEPTH-ANYTHING-V2</span>
            <span class="tag">OPEN3D</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        # Left panel — inputs
        with gr.Column(scale=1, min_width=300):
            image_input = gr.Image(
                label="Input Image",
                type="numpy",
                height=280,
            )
            classes_input = gr.Textbox(
                label="Object Classes",
                placeholder="chair, table, person",
                value="chair",
            )
            run_btn = gr.Button(
                "⬡  GENERATE POINT CLOUD",
                variant="primary",
                elem_classes=["primary-btn"]
            )
            gr.HTML("""
            <div style="
                margin-top:16px;
                padding:12px 16px;
                background:#0d0d0d;
                border:1px solid #1a1a1a;
                border-radius:8px;
                font-family:'Space Mono',monospace;
                font-size:10px;
                color:#444;
                line-height:1.8;
            ">
                <div style="color:#333;margin-bottom:6px">— HOW IT WORKS —</div>
                <div>① YOLO-World detects object</div>
                <div>② SAM segments exact pixels</div>
                <div>③ Depth-Anything estimates Z</div>
                <div>④ Back-projection → 3D points</div>
                <div>⑤ Metrics computed + denoised</div>
            </div>
            """)

        # Right panel — outputs
        with gr.Column(scale=2):
            viewer_output = gr.HTML(
                label="3D Point Cloud Viewer",
                value="""
                <div style="
                    height:600px;
                    background:#0d0d0d;
                    border:1px solid #1a1a1a;
                    border-radius:12px;
                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                    font-family:'Space Mono',monospace;
                    color:#2a2a2a;
                ">
                    <div style="font-size:48px;margin-bottom:16px">⬡</div>
                    <div style="font-size:12px;letter-spacing:2px">AWAITING INPUT</div>
                    <div style="font-size:10px;color:#1a1a1a;margin-top:8px">
                        Upload image and click generate
                    </div>
                </div>
                """
            )
            metrics_output = gr.HTML(label="Quality Metrics")

    run_btn.click(
        fn=process_image,
        inputs=[image_input, classes_input],
        outputs=[viewer_output, metrics_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        show_error=True,
        share=False
    )