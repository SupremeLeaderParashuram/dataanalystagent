# Data Analyst Agent API

A comprehensive API that uses AI and data science techniques to source, prepare, analyze, and visualize any data. This agent can handle various data analysis tasks including web scraping, database queries, statistical analysis, and visualization generation.

## Features

- **Web Scraping**: Automatically scrape data from web sources like Wikipedia
- **Database Integration**: Query large datasets using DuckDB with S3 support
- **Statistical Analysis**: Perform correlations, regressions, and other statistical computations
- **Data Visualization**: Generate plots and charts encoded as base64 data URIs
- **Multiple Data Formats**: Handle CSV, JSON, Parquet, and other data formats
- **RESTful API**: Simple POST endpoint for data analysis requests

## API Endpoint

The API exposes a single endpoint:

```
POST /api/
```

### Request Format

Send a POST request with multipart form data:
- `questions.txt` (required): Contains the analysis questions/tasks
- Additional files (optional): CSV, images, or other data files

### Example Usage

```bash
curl "https://your-api-endpoint.com/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv"
```

## Supported Analysis Types

### 1. Wikipedia Film Analysis
Scrapes highest-grossing films from Wikipedia and performs analysis:
- Count movies by criteria (year, revenue)
- Find earliest films meeting conditions
- Calculate correlations between rankings
- Generate scatter plots with regression lines

### 2. Indian High Court Judgments
Queries large-scale court judgment data:
- Count cases by court and time period
- Calculate processing delays and trends
- Generate trend visualizations
- Perform statistical analysis on legal data

### 3. Generic Data Analysis
- CSV data processing and analysis
- Statistical computations
- Custom visualizations
- Data aggregation and summarization

## Response Format

Responses are in JSON format and vary based on the analysis type:

### Array Response (for film analysis):
```json
[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]
```

### Object Response (for court analysis):
```json
{
  "Which high court disposed the most cases from 2019 - 2022?": "33_10",
  "What's the regression slope...?": 1.456789,
  "Plot the year and # of days...": "data:image/webp;base64,..."
}
```

## Installation and Deployment

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/data-analyst-agent.git
cd data-analyst-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t data-analyst-agent .
```

2. Run the container:
```bash
docker run -p 5000:5000 data-analyst-agent
```

### Cloud Deployment

This application can be deployed on various platforms:

- **Heroku**: Use the included `Procfile`
- **Google Cloud Run**: Deploy using the Dockerfile
- **AWS ECS/Fargate**: Container-based deployment
- **DigitalOcean App Platform**: Direct GitHub integration

## Environment Variables

- `PORT`: Port number for the application (default: 5000)
- `FLASK_ENV`: Flask environment (development/production)

## Dependencies

- **Flask**: Web framework
- **pandas**: Data manipulation and analysis
- **matplotlib/seaborn**: Data visualization
- **requests**: HTTP client for web scraping
- **BeautifulSoup**: HTML parsing
- **DuckDB**: Analytics database for large datasets
- **scipy**: Scientific computing and statistics
- **numpy**: Numerical computing

## Performance Considerations

- **Timeout**: All requests must complete within 5 minutes
- **Image Size**: Generated visualizations are kept under 100KB
- **Memory**: Efficient data processing for large datasets
- **Caching**: Session-based HTTP requests for web scraping

## Error Handling

The API includes comprehensive error handling:
- Invalid request formats return 400 errors
- Processing errors return 500 errors with error details
- Timeout protection for long-running analyses
- Fallback sample data for testing when external sources fail

## Testing

### Health Check
```bash
curl https://your-api-endpoint.com/health
```

### Sample Test
```bash
echo "Test analysis request" > questions.txt
curl "https://your-api-endpoint.com/api/" -F "questions.txt=@questions.txt"
```

## Security Considerations

- Input validation for all file uploads
- Safe HTML parsing to prevent injection attacks
- Resource limits to prevent DoS attacks
- No persistent data storage (stateless design)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the logs for detailed error information
- Verify your request format matches the examples

## Architecture

The application follows a modular design:
- `DataAnalystAgent`: Core analysis engine
- Flask routing for API endpoints
- Specialized methods for different data sources
- Efficient visualization generation with size optimization

## Roadmap

- [ ] Support for more data sources (APIs, databases)
- [ ] Advanced machine learning analysis
- [ ] Real-time data streaming
- [ ] Interactive dashboard generation
- [ ] Multi-language support for data processing
