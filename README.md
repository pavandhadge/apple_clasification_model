# Apple Health Detector

A Streamlit application to classify apples as "good" or "bad" from images or videos.

## Features

-   **Image and Video Classification:** Classify apples from both images and videos.
-   **Modern UI:** A clean and user-friendly interface built with Streamlit.
-   **Robust Model:** A model that is resilient to variations in lighting, like shadows and bright spots.
-   **Modular Codebase:** A well-organized and modular codebase for easy maintenance and extension.

## Project Structure

```
.
├── app.py
├── model
│   └── apple_model.h5
├── model.py
├── README.md
├── requirements.txt
├── train.py
├── ui.py
└── utils.py
```

-   `app.py`: The main entry point for the Streamlit application.
-   `ui.py`: Contains all the UI-related components.
-   `model.py`: Defines the model architecture and related functions (loading, prediction).
-   `train.py`: The script for training the model.
-   `utils.py`: Contains utility functions.
-   `requirements.txt`: Lists all the necessary dependencies.
-   `model/apple_model.h5`: The pre-trained model.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   On **macOS and Linux**:
        ```bash
        source venv/bin/activate
        ```
    -   On **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Application

To run the application, execute the following command in your terminal:

```bash
streamlit run app.py
```

### Training a New Model

To train a new model, you will need a dataset with the following structure:

```
dataset/
  train/
    good/
    bad/
  val/
    good/
    bad/
```

Once you have the dataset, you can train a new model by running the following command:

```bash
python train.py --dataset dataset
```

The trained model will be saved to `model/apple_model.h5`.
