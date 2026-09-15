# Food Detection AI

A web application for recognizing Vietnamese dishes on meal trays and calculating meal prices from uploaded photos or camera snapshots. The application combines a Flask backend, hierarchical CNN classification, and an interactive browser interface for selecting food regions.

Users position square crop boxes over individual dishes, submit the image for recognition, and review dish names, model scores, prices, and the total meal cost. A dedicated YOLO model checks for eggs when the CNN identifies braised pork, allowing the pricing logic to distinguish **Thịt kho** from **Thịt kho trứng**.

> **Model files required:** The supplied source archive contains six placeholder model files, each only 2 bytes long. They do not contain trained weights. Replace them with the actual trained models before running the application. The current archive cannot start the prediction server as supplied.

## Features

- **Image upload and camera capture:** Analyze an uploaded meal photo or take a snapshot from a browser-connected camera.
- **Interactive food regions:** Move and resize five square crop boxes, with two preset tray layouts. Crop settings are saved in the browser's `localStorage`.
- **Hierarchical classification:** Use a global CNN to identify food groups and specialist CNNs to classify dishes within selected groups.
- **Lighting preprocessing:** Apply conditional brightness correction, highlight compression, and mild CLAHE before CNN inference.
- **Egg-aware pricing:** Detect eggs in braised-pork crops and adjust the dish label and price accordingly.
- **Result dashboard:** Display crop thumbnails, predicted labels, model scores, up to five candidate predictions per region, and backend processing time.
- **Meal cost calculation:** Look up configured dish prices in Vietnamese đồng and sum the region prices.
- **Result export:** Download predictions and crop coordinates as JSON, or open the browser print dialog for a printable report.

Recognition is triggered by a button click. Camera preview is live, but the application does not continuously run inference on a video stream.

## How it works

1. The browser loads an image or captures a camera frame. The user adjusts the crop boxes to match the food regions.
2. The frontend sends the Base64 image and crop coordinates to `POST /predict`.
3. The backend clips each region to the image bounds, produces a square crop, applies conditional lighting preprocessing, and resizes the CNN input to **128 × 128** pixels. It converts BGR to RGB and scales pixel values to `[0, 1]`.
4. The global CNN selects the two highest-scoring food groups. For groups with a specialist model, candidate scores are calculated as `global group score × specialist class score`. Groups without a specialist model retain their global score.
5. The highest-scoring candidate determines the dish label. If it is **Thịt kho**, the YOLO model checks the original crop for eggs and the pricing rule may change the label to **Thịt kho trứng**.
6. The API returns each region's prediction, price, thumbnail, candidate scores, and lighting metadata, together with the total price and processing time.

The interface primarily labels the recognition system as CNN; the backend also uses YOLO for the braised-pork egg check. The reported percentages are model scores, not measured test accuracy or calibrated guarantees of correctness.

## Supported food labels

The code defines the following groups and output labels. Actual recognition requires models with matching output order.

| Global group | Dish labels |
| --- | --- |
| `Canh` | Canh chua có cá; Canh chua không cá; Canh rau cải thảo; Canh rau muống |
| `Chien` | Trứng chiên; Trứng chiên thịt |
| `Com` | Rice, represented as `Com` in the backend |
| `Khac` | Đậu hũ sốt cà |
| `Kho` | Cá hú kho; Thịt kho; Thịt kho trứng after the egg check |
| `Nuong` | Grilled dish, represented as `Nuong` in the backend |
| `Xao` | Rau xào củ sắn; Rau xào đậu dừa; Rau xào đậu que; Rau xào Lagim |

The internal group `Khac` maps to **Đậu hũ sốt cà**. It is not a general-purpose unknown-food rejection class.

## Technology stack

| Component | Technology |
| --- | --- |
| Backend and API | Python, Flask, Flask-CORS |
| CNN inference | TensorFlow / Keras |
| Egg detection | Ultralytics YOLO |
| Image processing | OpenCV, NumPy |
| Frontend | HTML, CSS, vanilla JavaScript |
| Browser features | Camera access, Canvas, localStorage, JSON download, printing |
| Additional declared dependency | Pillow |

The application does not require MySQL, a database service, or an external AI API key. The supplied code does not load a `.env` file.

## Project files

| Path | Purpose |
| --- | --- |
| `app.py` | Model loading, image preprocessing, inference, pricing, and Flask routes |
| `index.html` | Browser interface, camera capture, crop controls, result display, and export |
| `requirements.txt` | Python dependencies |
| `model/global2.h5` | Global CNN weights — must be supplied |
| `model/Canh_CNN_Best(v2).h5` | Soup classifier weights — must be supplied |
| `model/Chien_CNN_Best.h5` | Fried-dish classifier weights — must be supplied |
| `model/Kho_CNN_Best(v2).h5` | Braised-dish classifier weights — must be supplied |
| `model/Xao_CNN_Best(v2).h5` | Stir-fried-dish classifier weights — must be supplied |
| `model/model.pt` | YOLO egg-detector weights — must be supplied for egg detection |
| `README.md` | Project overview and usage instructions |
| `README_THAY_THE.txt` | Earlier replacement instructions included in the archive |

## Setup and local execution

### 1. Prepare the project and model weights

Download or clone the repository, then open a terminal in the folder containing `app.py` and `requirements.txt`.

Replace the placeholder files in `model/` with the original trained weights, keeping the filenames exactly as listed above. All five `.h5` files are loaded before Flask starts. The global model must use this output order:

```python
["Canh", "Chien", "Com", "Khac", "Kho", "Nuong", "Xao"]
```

The specialist models must match the class order in `SUB_MODELS` in `app.py`. The egg detector expects class ID `0` to represent eggs. The backend first checks `model/model.pt`, then falls back to `model.pt` beside `app.py`.

No training scripts, training dataset, model download links, or verified dependency versions are included. Use a Python/TensorFlow environment compatible with the original trained models; the repository does not establish a tested version combination.

### 2. Create a virtual environment

**Windows Command Prompt:**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The dependency list is currently unpinned. Installation alone does not verify compatibility with the trained weights.

### 4. Start the server

```bash
python app.py
```

After the models load successfully, open [http://localhost:5000](http://localhost:5000) in your browser. Flask serves both the interface and API from this address. Open the page through Flask for normal use.

To stop the server, press **Ctrl + C** in the terminal.

### Running in GitHub Codespaces

Supply the real model weights, create and activate the Linux virtual environment, install the dependencies, and run `python app.py` in the Codespaces terminal. Open forwarded port **5000** in the browser. This is a web application, so no VNC or desktop display is required.

The source archive does not include a `.devcontainer` configuration or an automated model download. A new Codespace still needs the setup above. Keep the server running while using the forwarded URL. See the [GitHub port-forwarding guide](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace) for opening and sharing the port.

## Using the application

1. Click **Tải ảnh** to upload a meal photo, or **Kết nối camera** and allow browser camera access.
2. Choose the tray preset closest to the image: three small regions above two large regions, or the reversed arrangement.
3. Drag the boxes over the dishes. Resize them using the handles; crops remain square. Use **Reset khung** to restore the preset.
4. Click **Bắt đầu nhận diện** or **Chụp / nhận diện**.
5. Review the predicted dishes, scores, individual prices, and total. Inspect the candidate predictions when a result looks uncertain.
6. Save the JSON result or use the report export action to open the print dialog.

The frontend processes five configured regions. The API accepts up to eight supplied boxes. There is no implemented empty-compartment detector: a blank or misplaced crop can still receive a food label and contribute to the price.

## Pricing rules

Prices are fixed project configuration in **VND**, not live market prices or estimates based on portion weight. Edit `PRICE_TABLE` and the egg-pricing constants in `app.py` to change them.

| Braised-pork result | Configured price |
| --- | --- |
| No eggs detected | 25,000 VND |
| One egg detected | 30,000 VND |
| More than one egg detected | 30,000 VND + 6,000 VND for each egg beyond the first |

Other dishes use `PRICE_TABLE`. The meal total is the sum of the processed region prices.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serve the browser interface |
| `GET` | `/health` | Return `{"status": "ok", "model": "CNN"}` after the server starts |
| `POST` | `/predict` | Classify food regions and calculate prices |

Example request using only Python's standard library, after the server is running and `meal.jpg` is available:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

image_b64 = base64.b64encode(Path("meal.jpg").read_bytes()).decode("ascii")
payload = {
    "image": "data:image/jpeg;base64," + image_b64,
    "boxes": [
        {
            "name": "Món 1",
            "x": 0.1,
            "y": 0.1,
            "w": 0.3,
            "h": 0.3,
            "unit": "normalized",
        }
    ],
}
request = Request(
    "http://localhost:5000/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request) as response:
    result = json.load(response)

for item in result["items"]:
    print(item["name"], item["final_class"], item["price"])
print("Total (VND):", result["total"])
```

Adjust the example coordinates to the food location. Normalized coordinates are fractions of image width and height; a rectangular pixel region is center-cropped to a square. Pixel coordinates are also accepted with `unit: "pixel"`. If no boxes are supplied, the backend uses five fixed regions selected by `layout: "top3"` or `"bottom3"`.

The response contains `items`, `total`, `processing_ms`, `model`, `source_size`, and `used_boxes`. Each item includes `name`, `final_class`, `raw_class`, `cnn_prob`, `price`, `top5`, `thumb_b64`, and `lighting`. `used_boxes` records clipped interactive regions and is empty for the fixed-layout fallback.

## Troubleshooting and limitations

| Symptom or limitation | Explanation / next step |
| --- | --- |
| Startup fails while loading an H5 file | Replace the 2-byte placeholders with real models. If valid models still fail, check compatibility with the original training environment. |
| Startup fails while loading `model.pt` | An existing invalid YOLO file is loaded without a protective exception handler. Replace it with valid weights. |
| YOLO package or weights are absent | With valid CNN weights, the backend can start without egg detection. Braised pork then receives the no-egg label and price; `/health` does not report this degraded state. |
| Camera is unavailable | Allow browser camera access and check device availability. Use image upload if camera access is blocked. |
| Incorrect dish or total | Check crop placement, lighting, and whether the food matches a supported class. Every processed crop contributes a predicted dish price. |
| Candidate list contains fewer than five dishes | Candidates come only from the two selected global groups; some group combinations produce fewer than five candidates. |
| History disappears on reload | Recent images are held in page memory, with only the last three displayed. Crop settings persist in `localStorage`; there is no database-backed history. |

The archive contains inference code but no evaluation dataset, benchmark report, or accuracy measurements. Lighting preprocessing is implemented, but its effect on recognition quality has not been established by the supplied files. Some sidebar entries and status badges are presentation elements rather than separate implemented pages or live diagnostics.

The application has no user authentication, payment processing, or persistent order storage. The current Flask startup command is intended for development and demonstration; a public production service requires additional deployment and access-control work.

## License

No `LICENSE` file is included in the supplied archive. Usage and redistribution terms for the code and model weights have not been specified.
