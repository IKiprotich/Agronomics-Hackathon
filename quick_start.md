# 🚀 Quick Start Guide

Get your Agricultural Dashboard running in 5 minutes!

## Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Generate Sample Data (Optional)
```bash
python generate_sample_data.py
```
This creates sample CSV files in the `sample_data/` folder.

## Step 3: Run the Dashboard
```bash
streamlit run agricultural_dashboard.py
```

## Step 4: Open in Browser
Navigate to `http://localhost:8501`

## Step 5: Start Analyzing!

### Option A: Use Sample Data
1. Go to "Data Input" tab
2. Select "Use Sample Data"
3. Click "Generate Sample Data"
4. Move to "Model Training" tab
5. Train your models and explore!

### Option B: Upload Your Own Data
1. Go to "Data Input" tab
2. Select "Upload Custom Data"
3. Upload your CSV files (see README for format)
4. Train models and analyze

### Option C: Quick Manual Test
1. Go to "Data Input" tab
2. Select "Manual Input"
3. Enter sample values
4. Click "Analyze Single Entry"

## Key Features to Try:

### 🤖 Machine Learning Models
- Train Random Forest for environmental prediction
- Build decision trees for effectiveness classification
- Perform K-means clustering

### 📊 Interactive Visualizations
- Effectiveness distribution charts
- County comparison plots
- Clustering visualizations
- Sankey flow diagrams

### 🔍 Advanced Analysis
- County performance rankings
- Crop/livestock comparisons
- Correlation analysis
- AI-generated recommendations

### 💡 Export Results
- Download analysis results as CSV
- Export summary reports
- Save interactive charts

## Sample Data Structure:

If you want to prepare your own data, use these formats:

### Environmental Data:
```csv
temperature,humidity,pressure,hour,month,day_of_year,day_of_week
22.5,65.3,840.2,12,6,150,1
21.8,68.1,838.5,13,6,151,2
```

### Crop Data:
```csv
county,crop,production,area_planted,yield_per_hectare
Nairobi,Maize,5000,500,2.5
Kiambu,Beans,3000,300,3.0
```

### Livestock Data:
```csv
county,livestock,population,productivity
Nairobi,Indigenous cattle,1000,1.2
Kiambu,Goats,2000,1.1
```

## Troubleshooting:

**Module not found error?**
```bash
pip install -r requirements.txt
```

**Dashboard won't load?**
Check that port 8501 is available, or use:
```bash
streamlit run agricultural_dashboard.py --server.port 8502
```

**Data upload issues?**
Make sure your CSV files match the expected format exactly.

## Next Steps:
1. Explore all 5 tabs of the dashboard
2. Try different model parameters in the sidebar
3. Compare multiple counties and crops
4. Generate and download reports
5. Use insights for agricultural planning

**Happy analyzing! 🌾📊** 