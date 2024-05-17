# AutoEDA-Dash

AutoEDA-Dash is a Plotly Dash application that performs automated Exploratory Data Analysis (EDA) on a dataset and displays the results in an interactive dashboard. This tool helps users quickly understand the structure, patterns, and relationships in their data without writing extensive code.

## Features

- Automated data loading and preprocessing
- Interactive visualizations using Plotly
- Statistical summaries and insights
- Customizable settings for visualizations

## Project Structure

    AutoEDA-Dash/
    │
    ├── .gitignore
    ├── README.md
    ├── app.py
    ├── exceptions.py
    └── requirements.txt

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/juan64aca/AutoEDA-Dash.git
    cd AutoEDA-Dash
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```sh
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application:**
    ```sh
    python app.py
    ```

2. **Access the dashboard:**
   Open your web browser and go to `http://127.0.0.1:8050/`.

3. **Upload your dataset:**
   Use the file upload feature in the dashboard to load your dataset. The application will automatically perform EDA and display the results.
r the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue in this repository.

