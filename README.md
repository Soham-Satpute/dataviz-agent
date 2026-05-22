# 📊 AI Data Visualization Agent

A Streamlit application that acts as your personal data visualization expert, powered by Groq LLMs. Upload a CSV dataset, ask questions in plain English, and the AI agent will analyze your data, generate appropriate visualizations, and score its own output quality — all in a polished dark-themed interface.

---

## Features

### 🤖 Conversational Data Analysis
- Ask questions about your data in plain English
- Full chat history with all past charts and results replayed on every rerun
- Interactive follow-up questioning with context retention
- AI-generated query suggestions based on your dataset's structure

### 📈 Intelligent Visualization
- Automatic selection of appropriate chart types
- Dynamic Python code generation and sandboxed local execution
- Matplotlib and Seaborn chart rendering
- Optional Plotly support (graceful fallback if not installed)

### 🔧 Data Preprocessing
- Normalize column names
- Remove duplicate rows
- Fill missing values
- Parse date-like columns
- Clip numeric outliers
- Download the preprocessed CSV at any time

### 📊 Dataset Insights Panel (right sidebar)
A live metrics panel with three sections that update as you work:

**① Dataset Quality**
- Data completeness (% non-missing cells)
- Row uniqueness (% non-duplicate rows)
- Overall health score

**② Data Structure**
- Numeric / categorical / datetime column breakdown
- Visualizable column count
- High-cardinality column detection (columns unsuitable for grouping)
- Skewed numeric column count
- Average outlier rate (values beyond 3σ)

**③ Output Quality** *(updates after each query)*
- **Execution success rate** — % of runs that produced output without errors
- **Column relevance** — % of code-referenced columns that actually exist; flags hallucinated column names
- **Data coverage** — % of dataset rows used in the result (detects `.head()` / `.sample()` underuse)
- **Chart type match** — rule-based check that the chosen chart type fits the query intent
- **Answer relevance** — LLM-as-judge score (1–5) with a one-sentence reason, powered by a lightweight Groq call

### ⚙️ Configuration
- Groq API key loaded from `.env` or entered per-session
- Model selection (Llama 3.3 70B recommended; any Groq-hosted model supported)
- API connection test button

### 📤 Export
- Download the latest analysis as a self-contained HTML report
- Full execution log table

---

## Setup

### 1. Get a Groq API Key
Sign up for free at [https://console.groq.com](https://console.groq.com) and create an API key.

## 2. Clone the repository
```bash
git clone https://github.com/Soham-Satpute/dataviz-agent.git
cd dataviz-agent
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key
Create a `.env` file in the project directory:
```
GROQ_API_KEY=your_key_here
```
Or paste it directly into the sidebar when the app is running.

### 5. Run the app
```bash
streamlit run ai_data_visualisation_agent.py
```

---

## Requirements

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `groq` | LLM inference (code generation + LLM-as-judge scoring) |
| `pandas` | Data loading, profiling, preprocessing |
| `matplotlib` | Chart rendering |
| `seaborn` | Statistical chart styles |
| `plotly` | Optional interactive charts (graceful fallback if missing) |
| `Pillow` | Reading generated chart images |
| `python-dotenv` | Loading `.env` API key |

---

## Supported Models (via Groq)

Any model available on your Groq account works. Recommended:

- **Llama 3.3 70B** — best overall quality for analysis and code generation
- **Llama 3.1 8B** — faster, lower latency for simpler queries
- **Mixtral 8x7B** — good balance of speed and quality
- **DeepSeek R1** — strong reasoning for complex statistical queries

---

## How It Works

1. Upload a CSV file via the sidebar
2. The app profiles the dataset (missing values, types, distributions, outliers)
3. Type a question in the chat input — the agent sends it to Groq with full dataset context
4. Groq returns Python code; the app validates and executes it in a sandboxed subprocess
5. Charts and text output are captured, displayed, and stored in chat history
6. A second lightweight Groq call scores the answer's relevance (1–5) for the Output Quality panel
7. All metrics and charts persist across reruns via Streamlit session state
