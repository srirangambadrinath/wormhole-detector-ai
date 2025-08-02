\# Wormhole Detector using Deep Spatio-Temporal Learning (CNN + Attention + Bi-GRU)



This project presents a novel AI-based framework for detecting astrophysical anomalies — specifically, \*\*wormhole candidates\*\* — from low-dimensional telescope-inspired inputs. The model autonomously classifies space-time signatures as:



\- `normal`

\- `blackhole`

\- `wormhole\_candidate`



---



\## 🧠 Model Architecture



This deep learning architecture combines:



\- \*\*Convolutional Neural Networks (CNN)\*\*: Extracts localized spatial features

\- \*\*Attention Layer\*\*: Focuses on important regions dynamically

\- \*\*Bidirectional GRU (Bi-GRU)\*\*: Captures forward and backward temporal dependencies



\### Architecture Flow:

Input (64x64x1)

↓

CNN Layers (128 filters)

↓

Flatten → Dense + Attention

↓

Attention Weighted Sum

↓

Bi-GRU Layer

↓

Output Dense (Softmax)



\## 🛰 Dataset



\- Dataset Name: \*\*Wormhole-SynthSpaceNet\*\*

\- Format: 64x64 grayscale `.png` images

\- Classes:

&nbsp; - `normal`: non-anomalous space

&nbsp; - `blackhole`: high-gravity collapse zones

&nbsp; - `wormhole\_candidate`: intense dual-lensed signatures



To generate it:

**Run the Pipeline:**



Step 1: Install Requirements



pip install -r requirements.txt



Step 2: Generate the Dataset



python generate\_data.py



Step 3: Train the Model



python train\_wormhole.py



Step 4: Evaluate and Visualize



python final.py

python generate\_data.py





**FOLDER STRUCTURE:**



wormhole\_detector/

├── data\_test/

│   ├── normal/

│   ├── blackhole/

│   └── wormhole\_candidate/

├── model\_wormhole.py

├── generate\_data.py

├── train\_wormhole.py

├── evaluate\_wormhole.py

├── final.py

├── temp\_weights.h5

├── wormhole\_model\_full.h5

├── training\_results.png

├── README.md

├── requirements.txt

└── .gitignore





