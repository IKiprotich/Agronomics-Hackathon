# 🌾 Agricultural Data Analytics Dashboard

An interactive web dashboard for analyzing agricultural data with **enhanced correlation analysis**, **Sankey flow diagrams**, and **clear effectiveness categorization** using advanced machine learning techniques.

## 🆕 Major Enhancements

### 🔗 **Comprehensive Correlation Analysis**
- **County-Crop/Livestock Correlation Matrix**: Visual heatmaps showing relationships between counties and agricultural products
- **Tabulated Results**: Color-coded summary tables with effectiveness rankings
- **Interactive Data Explorer**: Filter and explore data by county, type, and effectiveness level

### 🌊 **Sankey Flow Diagrams**
- **Flow Visualization**: Clear diagrams showing County → Crop/Livestock → Effectiveness relationships
- **Production Volume Representation**: Flow width represents production volume/population size
- **Color-coded Effectiveness**: Green (Highest), Yellow (Moderate), Red (Least Effective)

### 🎯 **Clear Effectiveness Categorization**
- **Highest Effectiveness**: Top 33% performers (🟢 Green indicators)
- **Moderate Effectiveness**: Middle 33% performers (🟡 Yellow indicators)
- **Least Effective**: Bottom 33% performers (🔴 Red indicators)
- **Enhanced Legends**: Clear legends on all visualizations and tables

## 🚀 Quick Start

### Option 1: Using the Runner Script (Recommended)
```bash
python run_dashboard.py
```
This script will:
- ✅ Install required packages automatically
- 🚀 Start the dashboard server
- 🔄 Provide restart options
- 🛠️ Handle server management

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run agricultural_dashboard.py
```

The dashboard will be available at: **http://localhost:8501**

## 📊 Features Overview

### **Data Input Options**
- **Upload Custom Data**: Upload your own CSV files for environmental, crop, and livestock data
- **Sample Data Generation**: Generate synthetic data for testing and demonstration
- **Manual Input**: Enter single data points for quick analysis

### **Machine Learning Models**
- **Random Forest**: Environmental suitability prediction with feature importance
- **Decision Tree**: Crop/livestock effectiveness classification
- **K-Means Clustering**: County-based agricultural clustering with PCA visualization

### **Enhanced Visualizations**
- **Correlation Heatmaps**: County vs Crop/Livestock effectiveness correlation
- **Sankey Diagrams**: Flow analysis from counties to crops to effectiveness levels
- **Interactive Charts**: Plotly-based charts with filtering and selection
- **Effectiveness Analysis**: Pie charts, bar charts, and heatmaps with clear legends
- **Data Explorer**: Multi-select filters with real-time updates

### **Advanced Analysis**
- **County Performance**: Ranking and metrics for each county with color coding
- **Crop/Livestock Analysis**: Performance comparison across different types
- **Effectiveness Factors**: Correlation analysis and success factors
- **Recommendations**: AI-generated actionable insights

### **Key Insights & Export**
- Automated summary of key findings
- Performance benchmarks and comparisons
- CSV export functionality for all analysis results
- Enhanced data tables with styling

## 🎨 Color Coding System

### Effectiveness Levels:
- **🟢 Highest (Green #28a745)**: Top performing counties/crops (33rd+ percentile)
- **🟡 Moderate (Yellow #ffc107)**: Average performance (33rd-67th percentile)
- **🔴 Least Effective (Red #dc3545)**: Needs improvement (0-33rd percentile)

### Visual Elements:
- **Tables**: Background colors match effectiveness levels
- **Charts**: Consistent color scheme across all visualizations
- **Legends**: Clear indication of what each color represents

## 📈 Dashboard Tabs

### 1. **📊 Data Input Tab**
Choose your data input method and preview your data:
- Upload CSV files or use sample data
- Preview data structure and quality
- Manual input for quick single-entry analysis

### 2. **🤖 Model Training Tab**
Train and evaluate machine learning models:
- Environmental prediction models with R² scores
- Effectiveness classification with accuracy metrics
- Clustering analysis with silhouette scores

### 3. **📈 Interactive Visualizations Tab** (Enhanced)
#### **Correlation Analysis Section**
- County-Type correlation matrix heatmap
- County-Category correlation visualization
- Styled summary tables with effectiveness rankings
- Download correlation data as CSV

#### **Sankey Diagrams Section**
- Flow visualization: County → Crop/Livestock → Effectiveness
- Production volume representation through flow width
- Clear interpretation guide with legend

#### **Effectiveness Analysis Section**
- Overall effectiveness distribution pie chart
- County effectiveness stacked bar charts
- Crop/Livestock effectiveness heatmaps
- Detailed analysis tables with highlighting

#### **Interactive Data Explorer Section**
- Multi-select filters for counties, types, and effectiveness levels
- Real-time scatter plots with hover information
- Filtered summary statistics

### 4. **🔍 Analysis Tab**
Detailed performance analysis:
- County performance rankings
- Crop/livestock effectiveness comparisons
- Success factor identification
- Actionable recommendations with priority levels

### 5. **💡 Insights Tab**
Key statistics and summary:
- Overall effectiveness percentages
- Best/worst performing counties
- Category comparisons
- Export functionality for reports

## 📋 Data Structure

### Environmental Data CSV:
```csv
temperature,humidity,pressure,hour,month,day_of_year,day_of_week
22.5,65.3,840.2,12,6,150,1
23.1,68.7,835.8,13,6,151,2
```

### Crop Data CSV:
```csv
county,crop,production,area_planted,yield_per_hectare
Nairobi,Maize,5000,500,2.5
Kiambu,Beans,3000,400,1.8
Nakuru,Potatoes,8000,600,3.2
```

### Livestock Data CSV:
```csv
county,livestock,population,productivity
Nairobi,Indigenous cattle,1000,1.2
Kiambu,Goats,2500,1.4
Meru,Sheep,1800,1.1
```

## 🛠️ Server Management

### Starting the Server:
```bash
# Using runner script (recommended)
python run_dashboard.py

# Or directly
streamlit run agricultural_dashboard.py
```

### Restarting the Server:
- Press `Ctrl+C` to stop the server
- Run the script again, or
- Use the runner script's restart option

### Server Configuration:
- **Default port**: 8501
- **URL**: http://localhost:8501
- **Network access**: Configurable for remote access

## 📊 Visualization Features

### Chart Types:
- **Correlation Heatmaps**: Plotly-based interactive heatmaps
- **Sankey Diagrams**: Flow diagrams with customizable colors
- **Pie Charts**: Effectiveness distribution with percentages
- **Bar Charts**: County and type comparisons
- **Scatter Plots**: Interactive filtering and selection
- **Heatmaps**: Type vs effectiveness distribution

### Interactive Elements:
- **Hover Information**: Detailed data on hover
- **Multi-select Filters**: Real-time data filtering
- **Download Options**: Export charts and data
- **Responsive Design**: Works on different screen sizes

## 🔧 Technical Details

### Built With:
- **Streamlit**: Web dashboard framework
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning models
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Tabulate**: Enhanced data table formatting

### Machine Learning Pipeline:
- **Data Preprocessing**: Automated cleaning and feature engineering
- **Model Training**: Cross-validation and hyperparameter tuning
- **Evaluation**: Multiple metrics (R², accuracy, silhouette score)
- **Prediction**: Real-time effectiveness classification

## 📥 Export Capabilities

### Available Exports:
- **County Summary Data**: CSV format with all metrics
- **Detailed Effectiveness Analysis**: Comprehensive analysis tables
- **Correlation Data**: Complete correlation matrices
- **Filtered Results**: Export filtered datasets

### Export Formats:
- **CSV**: For data analysis and further processing
- **Charts**: PNG/HTML format via Plotly
- **Reports**: Summary insights and recommendations

## 🚨 Troubleshooting

### Common Issues:

1. **Dependencies**: Install all required packages
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**: Change port or stop existing processes
   ```bash
   # Kill process using port 8501
   lsof -ti:8501 | xargs kill -9
   ```

3. **Data Format**: Ensure CSV files match expected structure

4. **Memory Issues**: Use sample data for large datasets

5. **Visualization Not Loading**: Check Plotly installation
   ```bash
   pip install plotly --upgrade
   ```

### Performance Tips:
- Use sample data for initial testing
- Filter data when working with large datasets
- Restart server if experiencing memory issues
- Close unused browser tabs to free up memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or support:
- Check the troubleshooting section
- Review the data structure requirements
- Ensure all dependencies are installed
- Verify CSV file formats match expectations

---

**🌾 Happy Analyzing! Transform your agricultural data into actionable insights with enhanced correlation analysis, beautiful Sankey diagrams, and clear effectiveness categorization.** 