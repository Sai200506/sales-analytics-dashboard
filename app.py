import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_and_validate_data(uploaded_file):
    """Load and validate the uploaded sales data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a CSV or Excel file")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def clean_and_prepare_data(df):
    """Clean and prepare the sales data for analysis"""
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Common column name mappings
    column_mappings = {
        'date': ['date', 'order_date', 'sale_date', 'transaction_date'],
        'sales': ['sales', 'revenue', 'amount', 'total', 'sales_amount'],
        'quantity': ['quantity', 'qty', 'units', 'units_sold'],
        'product': ['product', 'product_name', 'item', 'product_category'],
        'customer': ['customer', 'customer_name', 'client', 'customer_id'],
        'region': ['region', 'location', 'city', 'state', 'country']
    }
    
    # Standardize column names
    df_columns_lower = [col.lower().replace(' ', '_') for col in df_clean.columns]
    df_clean.columns = df_columns_lower
    
    # Try to identify and rename important columns
    renamed_columns = {}
    for standard_name, possible_names in column_mappings.items():
        for col in df_clean.columns:
            if any(name in col.lower() for name in possible_names):
                renamed_columns[col] = standard_name
                break
    
    df_clean.rename(columns=renamed_columns, inplace=True)
    
    # Convert date column if exists
    if 'date' in df_clean.columns:
        try:
            df_clean['date'] = pd.to_datetime(df_clean['date'])
        except:
            st.warning("Could not convert date column. Please ensure date format is valid.")
    
    # Convert numeric columns
    numeric_columns = ['sales', 'quantity']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def calculate_key_metrics(df):
    """Calculate key sales metrics"""
    metrics = {}
    
    if 'sales' in df.columns:
        metrics['total_revenue'] = df['sales'].sum()
        metrics['avg_sales_per_transaction'] = df['sales'].mean()
        metrics['median_sales'] = df['sales'].median()
    
    if 'quantity' in df.columns:
        metrics['total_units_sold'] = df['quantity'].sum()
        metrics['avg_quantity_per_transaction'] = df['quantity'].mean()
    
    metrics['total_transactions'] = len(df)
    
    if 'customer' in df.columns:
        metrics['unique_customers'] = df['customer'].nunique()
    
    if 'product' in df.columns:
        metrics['unique_products'] = df['product'].nunique()
    
    return metrics

def create_time_series_analysis(df):
    """Create time series analysis charts"""
    if 'date' not in df.columns or 'sales' not in df.columns:
        return None
    
    # Aggregate sales by date
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    
    # Create time series plot
    fig = px.line(daily_sales, x='date', y='sales', 
                  title='Sales Trend Over Time',
                  labels={'sales': 'Sales ($)', 'date': 'Date'})
    fig.update_layout(height=400)
    
    return fig

def create_product_analysis(df):
    """Analyze product performance"""
    if 'product' not in df.columns or 'sales' not in df.columns:
        return None, None
    
    # Top products by sales
    product_sales = df.groupby('product')['sales'].sum().sort_values(ascending=False).head(10)
    
    fig1 = px.bar(x=product_sales.values, y=product_sales.index, 
                  orientation='h', title='Top 10 Products by Sales',
                  labels={'x': 'Sales ($)', 'y': 'Product'})
    fig1.update_layout(height=400)
    
    # Product sales distribution
    fig2 = px.pie(values=product_sales.head(5).values, 
                  names=product_sales.head(5).index,
                  title='Top 5 Products Sales Distribution')
    
    return fig1, fig2

def create_customer_analysis(df):
    """Analyze customer behavior"""
    if 'customer' not in df.columns or 'sales' not in df.columns:
        return None, None
    
    # Customer spending analysis
    customer_sales = df.groupby('customer')['sales'].agg(['sum', 'count']).reset_index()
    customer_sales.columns = ['customer', 'total_sales', 'transaction_count']
    customer_sales = customer_sales.sort_values('total_sales', ascending=False).head(10)
    
    fig1 = px.bar(customer_sales, x='customer', y='total_sales',
                  title='Top 10 Customers by Total Sales',
                  labels={'total_sales': 'Total Sales ($)', 'customer': 'Customer'})
    fig1.update_layout(height=400)
    
    # Customer transaction frequency
    fig2 = px.scatter(customer_sales, x='transaction_count', y='total_sales',
                      hover_data=['customer'],
                      title='Customer Value vs Transaction Frequency',
                      labels={'transaction_count': 'Number of Transactions', 
                             'total_sales': 'Total Sales ($)'})
    
    return fig1, fig2

def generate_insights_and_recommendations(df, metrics):
    """Generate business insights and recommendations"""
    insights = []
    recommendations = []
    
    # Revenue insights
    if 'total_revenue' in metrics:
        insights.append(f"💰 Total Revenue: ${metrics['total_revenue']:,.2f}")
        insights.append(f"📊 Average Transaction Value: ${metrics['avg_sales_per_transaction']:,.2f}")
    
    # Customer insights
    if 'unique_customers' in metrics and 'total_transactions' in metrics:
        avg_transactions_per_customer = metrics['total_transactions'] / metrics['unique_customers']
        insights.append(f"👥 Average Transactions per Customer: {avg_transactions_per_customer:.1f}")
        
        if avg_transactions_per_customer < 2:
            recommendations.append("🎯 Focus on customer retention strategies to increase repeat purchases")
    
    # Product insights
    if 'product' in df.columns:
        product_performance = df.groupby('product')['sales'].sum().sort_values(ascending=False)
        top_product_share = product_performance.iloc[0] / product_performance.sum() * 100
        insights.append(f"🏆 Top product accounts for {top_product_share:.1f}% of total sales")
        
        if top_product_share > 50:
            recommendations.append("⚠️ Consider diversifying product portfolio to reduce dependency on single product")
        
        # Low performing products
        bottom_20_percent = product_performance.quantile(0.2)
        low_performers = product_performance[product_performance <= bottom_20_percent]
        if len(low_performers) > 0:
            recommendations.append(f"📉 Review {len(low_performers)} underperforming products for potential optimization or discontinuation")
    
    # Seasonal trends
    if 'date' in df.columns and 'sales' in df.columns:
        df['month'] = df['date'].dt.month
        monthly_sales = df.groupby('month')['sales'].mean()
        peak_month = monthly_sales.idxmax()
        low_month = monthly_sales.idxmin()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        insights.append(f"📅 Peak sales month: {month_names[peak_month-1]}")
        insights.append(f"📅 Lowest sales month: {month_names[low_month-1]}")
        
        recommendations.append(f"🚀 Plan marketing campaigns around {month_names[peak_month-1]} for maximum impact")
        recommendations.append(f"💡 Develop strategies to boost sales during {month_names[low_month-1]}")
    
    return insights, recommendations

# Main Streamlit app
def main():
    st.markdown('<h1 class="main-header">📊 Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Upload your sales data and get instant insights to boost your business performance!**
    
    Supported formats: CSV, Excel (.xlsx, .xls)
    """)
    
    # Sidebar
    st.sidebar.title("📈 Dashboard Controls")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Sales Data", 
        type=['csv', 'xlsx', 'xls'],
        help="Upload your sales data file to begin analysis"
    )
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Loading and processing data..."):
            df_raw = load_and_validate_data(uploaded_file)
            
            if df_raw is not None:
                df = clean_and_prepare_data(df_raw)
                
                st.sidebar.success(f"✅ Data loaded successfully!")
                st.sidebar.info(f"📋 Rows: {len(df)}, Columns: {len(df.columns)}")
                
                # Show data preview
                with st.expander("📋 Data Preview", expanded=False):
                    st.dataframe(df.head(10))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Dataset Info:**")
                        st.write(f"- Total rows: {len(df)}")
                        st.write(f"- Total columns: {len(df.columns)}")
                    with col2:
                        st.write("**Column Names:**")
                        st.write(list(df.columns))
                
                # Calculate metrics
                metrics = calculate_key_metrics(df)
                
                # Key Metrics Section
                st.markdown("## 🎯 Key Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'total_revenue' in metrics:
                        st.metric("Total Revenue", f"${metrics['total_revenue']:,.0f}")
                    
                with col2:
                    if 'total_transactions' in metrics:
                        st.metric("Total Transactions", f"{metrics['total_transactions']:,}")
                
                with col3:
                    if 'unique_customers' in metrics:
                        st.metric("Unique Customers", f"{metrics['unique_customers']:,}")
                
                with col4:
                    if 'avg_sales_per_transaction' in metrics:
                        st.metric("Avg Transaction Value", f"${metrics['avg_sales_per_transaction']:,.2f}")
                
                # Charts Section
                st.markdown("## 📊 Sales Analysis")
                
                # Time Series Analysis
                time_chart = create_time_series_analysis(df)
                if time_chart:
                    st.plotly_chart(time_chart, use_container_width=True)
                
                # Product and Customer Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    product_chart1, product_chart2 = create_product_analysis(df)
                    if product_chart1:
                        st.plotly_chart(product_chart1, use_container_width=True)
                
                with col2:
                    customer_chart1, customer_chart2 = create_customer_analysis(df)
                    if customer_chart1:
                        st.plotly_chart(customer_chart1, use_container_width=True)
                
                # Additional charts in second row
                col1, col2 = st.columns(2)
                
                with col1:
                    if product_chart2:
                        st.plotly_chart(product_chart2, use_container_width=True)
                
                with col2:
                    if customer_chart2:
                        st.plotly_chart(customer_chart2, use_container_width=True)
                
                # Insights and Recommendations
                st.markdown("## 💡 Business Insights & Recommendations")
                
                insights, recommendations = generate_insights_and_recommendations(df, metrics)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Key Insights")
                    for insight in insights:
                        st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 🎯 Recommendations")
                    for recommendation in recommendations:
                        st.markdown(f"<div class='insight-box'>{recommendation}</div>", unsafe_allow_html=True)
                
                # Download processed data
                st.markdown("## 📥 Download Results")
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data as CSV",
                    data=csv,
                    file_name="processed_sales_data.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("👆 Please upload a sales data file to begin analysis")
        
        # Sample data format
        st.markdown("## 📋 Expected Data Format")
        
        sample_data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Product': ['Product A', 'Product B', 'Product A'],
            'Customer': ['Customer 1', 'Customer 2', 'Customer 1'],
            'Sales': [1000, 1500, 800],
            'Quantity': [10, 15, 8],
            'Region': ['North', 'South', 'North']
        })
        
        st.dataframe(sample_data)
        
        st.markdown("""
        **Your data should include columns such as:**
        - **Date/Order Date**: Transaction date
        - **Sales/Revenue/Amount**: Sales amount in currency
        - **Product/Product Name**: Product identifier
        - **Customer/Customer Name**: Customer identifier
        - **Quantity/Units**: Number of units sold
        - **Region/Location**: Geographic information (optional)
        
        The dashboard will automatically detect and map your column names!
        """)

if __name__ == "__main__":
    main()
