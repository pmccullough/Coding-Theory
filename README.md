# Classical Coding Theory Project

This project explores the mathematical foundations and implementation of classical error-correcting codes, with a focus on BCH codes and decoding algorithms like the Peterson–Gorenstein–Zierler (PGZ) decoder.

## 📁 Project Structure

```
coding-theory-project/
├── src/          # Python code for encoding/decoding
├── notebooks/    # Jupyter notebooks for demos and testing
├── paper/        # LaTeX files for write-up
├── requirements.txt
├── README.md
└── venv/         # Python virtual environment (not tracked by Git)
```

## 🧠 Goals

- Understand the structure of linear, cyclic, and BCH codes
- Implement the Key Equation decoding algorithm
- Visualize decoding via Jupyter notebooks

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Set up your virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   .\venv\Scripts\activate       # Windows
   pip install -r requirements.txt
   ```

3. Launch VS Code:
   ```bash
   code .
   ```

4. Compile LaTeX:
   - Navigate to the `paper/` directory
   - Use VS Code with LaTeX Workshop or run `pdflatex main.tex`

## 🛠️ Dependencies

- Python 3.10+
- `numpy`
- `jupyter`
- LaTeX distribution (e.g., TeX Live, MikTeX) for compiling the report


