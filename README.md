# Best Buy Review Analyzer 🔍

An AI-powered comprehensive tool for scraping, analyzing, and generating insights from Best Buy product reviews using advanced machine learning and natural language processing techniques.

## 🌟 Features

### Core Functionality
- **Automated Web Scraping**: Parallel scraping of Best Buy product reviews using Selenium WebDriver
- **AI-Powered Topic Extraction**: Uses local Mistral LLM via Ollama for intelligent topic generation
- **Smart Clustering**: Automated optimal cluster determination using the elbow method with KneeLocator
- **Sentiment Analysis**: Advanced review sentiment classification and rating analysis
- **Data Visualization**: Rich charts and graphs using Matplotlib, Seaborn, and Plotly
- **Business Reporting**: Automated Excel and PowerPoint report generation

### Advanced Analytics
- **Vector Embeddings**: AI-generated embeddings for semantic review analysis
- **Cluster Optimization**: Fuzzy matching for merging similar clusters
- **Pareto Analysis**: Identification of top problem areas affecting customer satisfaction
- **Executive Summaries**: AI-generated concise problem explanations and recommendations

### Export Formats
- **CSV Files**: Raw and processed review data
- **Excel Reports**: Interactive spreadsheets with embedded charts and analysis
- **PowerPoint Presentations**: Professional business presentations with visualizations
- **Image Exports**: High-quality charts and graphs for external use

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with the following packages:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn
   pip install selenium webdriver-manager beautifulsoup4 requests
   pip install openpyxl python-pptx tqdm kneed fuzzywuzzy python-levenshtein
   pip install ollama
   ```

2. **Chrome/Edge WebDriver**: Automatically managed by webdriver-manager

3. **Ollama Setup**:
   ```bash
   # Install Ollama (https://ollama.ai)
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

### Installation

1. Clone or download the project files
2. Ensure `Project.ipynb` is in your working directory
3. Install all required dependencies
4. Start Ollama service:
   ```bash
   ollama serve
   ```

### Usage

1. **Configure Settings**: Update the configuration cell in the notebook:
   ```python
   # Configuration
   DRIVER_PATH = "auto"  # Auto-download ChromeDriver
   MAX_REVIEWS = 100     # Maximum reviews to scrape
   RETRY_ATTEMPTS = 3    # Retry failed requests
   REQUEST_TIMEOUT = 30  # Timeout in seconds
   ```

2. **Set Product URL**: Replace the Best Buy product URL in the notebook:
   ```python
   product_url = "https://www.bestbuy.com/site/your-product-url"
   ```

3. **Run the Notebook**: Execute all cells in Jupyter Notebook or JupyterLab

## 📊 Output Structure

The tool creates organized output directories:

```
C:\Path\
├── Project.ipynb           # Main analysis notebook
├── README.md              # This file
├── outputs/
│   ├── csv/               # Raw and processed data
│   │   ├── raw_reviews.csv
│   │   ├── processed_reviews.csv
│   │   └── clustered_reviews.csv
│   ├── excel/             # Excel reports with charts
│   │   └── bestbuy_analysis_report.xlsx
│   ├── images/            # Generated visualizations
│   │   ├── rating_distribution.png
│   │   ├── problem_clusters.png
│   │   └── pareto_analysis.png
│   └── presentations/     # PowerPoint reports
       └── bestbuy_analysis_presentation.pptx
```

## 🔧 Technical Architecture

### Data Pipeline
1. **Web Scraping** → Selenium-based parallel scraping
2. **Data Cleaning** → Duplicate removal, text normalization
3. **AI Processing** → Topic extraction using Mistral LLM
4. **Vectorization** → Semantic embeddings via Ollama
5. **Clustering** → Automated KMeans with optimal k-selection
6. **Analysis** → Pareto analysis and problem identification
7. **Reporting** → Multi-format export with visualizations

### AI Integration
- **Local LLM**: Mistral model for topic generation and summarization
- **Embedding Model**: Nomic-embed-text for semantic vector creation
- **Ollama API**: Local inference server for privacy and performance

### Key Technologies
- **Web Scraping**: Selenium WebDriver with Chrome/Edge
- **Data Processing**: Pandas, NumPy for data manipulation
- **Machine Learning**: Scikit-learn for clustering and analysis
- **Visualization**: Matplotlib, Seaborn, Plotly for charts
- **AI/NLP**: Ollama, Mistral LLM for text processing
- **Export**: OpenPyXL, python-pptx for business reports

## 📈 Sample Analysis Output

### Review Statistics
- Total reviews scraped and processed
- Rating distribution analysis
- Review date range and frequency
- Data quality metrics

### Problem Identification
- Top customer complaint categories
- Cluster analysis with AI-generated names
- Pareto chart showing 80/20 problem distribution
- Severity rankings and frequency counts

### Business Insights
- Executive summary of key findings
- Recommended action items
- Customer satisfaction trends
- Product improvement suggestions

## 🛠️ Configuration Options

### Scraping Settings
```python
MAX_REVIEWS = 100          # Maximum reviews to collect
RETRY_ATTEMPTS = 3         # Number of retry attempts for failed requests
REQUEST_TIMEOUT = 30       # HTTP request timeout in seconds
PARALLEL_WORKERS = 4       # Number of parallel scraping threads
```

### AI Model Settings
```python
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # Ollama API endpoint
LLM_MODEL = "mistral"                        # Language model for topics
EMBEDDING_MODEL = "nomic-embed-text"         # Embedding model
MAX_TOKENS = 150                             # Max tokens for AI responses
```

### Analysis Parameters
```python
MIN_CLUSTER_SIZE = 5       # Minimum reviews per cluster
MAX_CLUSTERS = 15          # Maximum number of clusters
SIMILARITY_THRESHOLD = 0.8 # Cluster merging threshold
```

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   ```bash
   # Start Ollama service
   ollama serve
   
   # Verify models are installed
   ollama list
   ```

2. **WebDriver Issues**:
   - Chrome/Edge browser must be installed
   - webdriver-manager handles driver downloads automatically
   - Check Chrome version compatibility

3. **Memory Issues**:
   - Reduce `MAX_REVIEWS` for large datasets
   - Adjust `PARALLEL_WORKERS` based on system resources

4. **Slow Performance**:
   - Use GPU acceleration for Ollama if available
   - Reduce embedding dimensions
   - Enable result caching

### Error Handling

The notebook includes comprehensive error handling:
- Automatic retry for failed web requests
- Graceful degradation when AI services are unavailable
- Progress tracking with detailed logging
- Data validation and quality checks

## 📝 Customization

### Adding New Analysis Features
1. **Custom Metrics**: Add new analysis functions in the processing section
2. **Visualization Types**: Create additional chart types using Plotly/Matplotlib
3. **Export Formats**: Add new output formats (PDF, Word, etc.)
4. **AI Models**: Experiment with different Ollama models

### Extending to Other Platforms
The architecture can be adapted for other e-commerce platforms:
- Modify scraping logic for different review structures
- Update CSS selectors for new websites
- Adjust data parsing for different review formats

## 📚 Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Static visualizations
- **plotly**: Interactive visualizations

### Web Scraping
- **selenium**: Web browser automation
- **webdriver-manager**: Automatic driver management
- **beautifulsoup4**: HTML parsing
- **requests**: HTTP library

### AI/ML
- **ollama**: Local LLM inference
- **kneed**: Automatic elbow detection
- **fuzzywuzzy**: Fuzzy string matching

### Export/Reporting
- **openpyxl**: Excel file generation
- **python-pptx**: PowerPoint creation
- **tqdm**: Progress bars

## 🤝 Contributing

This is an internship project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is developed as part of an internship program. Please check with the organization for usage rights and licensing terms.

## 🎯 Future Enhancements

- **Multi-platform Support**: Extend to Amazon, Walmart, etc.
- **Real-time Monitoring**: Scheduled analysis updates
- **Advanced NLP**: Aspect-based sentiment analysis
- **Interactive Dashboard**: Web-based visualization interface
- **API Integration**: RESTful API for programmatic access
- **Cloud Deployment**: Azure/AWS integration for scalability

## 📞 Support

For technical support or questions about this project:
- Review the troubleshooting section
- Check Ollama documentation for AI model issues
- Verify all dependencies are correctly installed
- Ensure Best Buy URLs are valid and accessible

---

**Built with ❤️ using Python, AI, and modern data science techniques.**
