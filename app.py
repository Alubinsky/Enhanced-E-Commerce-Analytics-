import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our analysis module
from ecommerce_analytics import EcommerceAnalyzer, EnhancedDecisionTreeAnalyzer

# Page configuration
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Sidebar
st.sidebar.title("🛍️ E-Commerce Analytics")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Navigate to",
    ["🏠 Overview", "👥 Customer Intelligence", "📊 Sales Analytics", 
     "🔍 Anomaly Detection", "📈 Forecasting", "🌍 Market Analysis",
     "📦 Product Analysis", "🌳 Decision Trees", "📋 Reports"]
)

# Load data button
if st.sidebar.button("🔄 Load/Refresh Data", type="primary"):
    with st.spinner("Loading data..."):
        analyzer = EcommerceAnalyzer()
        analyzer.load_data(fetch_from_uci=True)
        st.session_state.analyzer = analyzer
        st.session_state.data_loaded = True
        st.success("✅ Data loaded successfully!")

# Main content
st.markdown('<h1 class="main-header">🛍️ E-Commerce Analytics Dashboard</h1>', 
            unsafe_allow_html=True)

if not st.session_state.data_loaded:
    st.info("👈 Please click 'Load/Refresh Data' in the sidebar to begin")
    st.stop()

analyzer = st.session_state.analyzer

# ============================================================================
# OVERVIEW PAGE
# ============================================================================

if page == "🏠 Overview":
    st.header("Business Overview")
    
    # Calculate KPIs
    total_revenue = analyzer.df['Revenue'].sum()
    total_orders = analyzer.df['InvoiceNo'].nunique()
    total_customers = analyzer.df['CustomerID'].nunique()
    avg_order_value = analyzer.df.groupby('InvoiceNo')['Revenue'].sum().mean()
    
    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.0f}", "↑ 12%")
    with col2:
        st.metric("Total Orders", f"{total_orders:,}", "↑ 8%")
    with col3:
        st.metric("Total Customers", f"{total_customers:,}", "↑ 15%")
    with col4:
        st.metric("Avg Order Value", f"${avg_order_value:.2f}", "↑ 5%")
    
    st.markdown("---")
    
    # Revenue over time
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Revenue Trend")
        daily_revenue = analyzer.df.groupby(
            analyzer.df['InvoiceDate'].dt.date
        )['Revenue'].sum().reset_index()
        daily_revenue.columns = ['Date', 'Revenue']
        
        fig = px.line(daily_revenue, x='Date', y='Revenue',
                     title='Daily Revenue Trend')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Revenue by Country")
        country_revenue = analyzer.df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(x=country_revenue.values, y=country_revenue.index,
                    orientation='h', title='Top 10 Countries by Revenue')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent transactions
    st.subheader("📋 Recent Transactions")
    recent_data = analyzer.df.nlargest(100, 'InvoiceDate')[
        ['InvoiceDate', 'InvoiceNo', 'CustomerID', 'Country', 'Revenue']
    ].head(10)
    st.dataframe(recent_data, use_container_width=True)

# ============================================================================
# CUSTOMER INTELLIGENCE PAGE
# ============================================================================

elif page == "👥 Customer Intelligence":
    st.header("Customer Intelligence")
    
    tabs = st.tabs(["RFM Analysis", "Clustering", "Churn Prediction", "CLV"])
    
    with tabs[0]:  # RFM Analysis
        st.subheader("RFM Segmentation")
        
        if analyzer.customer_features is None:
            with st.spinner("Performing RFM analysis..."):
                analyzer.perform_rfm_analysis()
        
        # Segment distribution
        segment_counts = analyzer.customer_features['Segment'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                        title='Customer Segments Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # RFM Score distribution
            fig = px.scatter_3d(analyzer.customer_features, 
                              x='Recency', y='Frequency', z='Monetary',
                              color='Segment', title='RFM 3D Visualization',
                              height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment details
        st.subheader("Segment Performance")
        segment_summary = analyzer.customer_features.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerID': 'count'
        }).round(2)
        segment_summary.columns = ['Avg Recency (days)', 'Avg Frequency', 
                                   'Avg Monetary ($)', 'Customer Count']
        st.dataframe(segment_summary, use_container_width=True)
    
    with tabs[1]:  # Clustering
        st.subheader("Customer Clustering")
        
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        
        if st.button("Run Clustering"):
            with st.spinner("Performing clustering..."):
                analyzer.perform_clustering(n_clusters=n_clusters)
                st.success("✅ Clustering complete!")
        
        if 'Cluster' in analyzer.customer_features.columns:
            # Cluster visualization
            fig = px.scatter(analyzer.customer_features, 
                           x='Frequency', y='Monetary',
                           color='Cluster', size='Recency',
                           title='Customer Clusters',
                           hover_data=['CustomerID'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:  # Churn Prediction
        st.subheader("Churn Prediction")
        
        if st.button("Train Churn Model"):
            with st.spinner("Training churn prediction model..."):
                analyzer.predict_churn()
                st.success("✅ Churn model trained!")
        
        if 'Churn_Probability' in analyzer.customer_features.columns:
            # Churn risk distribution
            fig = px.histogram(analyzer.customer_features, 
                             x='Churn_Probability',
                             title='Churn Risk Distribution',
                             nbins=50)
            st.plotly_chart(fig, use_container_width=True)
            
            # High-risk customers
            high_risk = analyzer.customer_features[
                analyzer.customer_features['Churn_Probability'] > 0.7
            ].sort_values('Monetary', ascending=False).head(10)
            
            st.subheader("⚠️ High-Risk High-Value Customers")
            st.dataframe(high_risk[['CustomerID', 'Recency', 'Frequency', 
                                   'Monetary', 'Churn_Probability']], 
                        use_container_width=True)
    
    with tabs[3]:  # CLV
        st.subheader("Customer Lifetime Value")
        
        if st.button("Calculate CLV"):
            with st.spinner("Calculating CLV..."):
                analyzer.predict_clv()
                st.success("✅ CLV calculated!")
        
        if 'Predicted_CLV' in analyzer.customer_features.columns:
            # CLV distribution
            fig = px.box(analyzer.customer_features, 
                        y='Predicted_CLV',
                        title='CLV Distribution',
                        points="outliers")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top customers by CLV
            top_clv = analyzer.customer_features.nlargest(10, 'Predicted_CLV')
            st.subheader("🌟 Top Customers by CLV")
            st.dataframe(top_clv[['CustomerID', 'Predicted_CLV', 'Frequency', 'Monetary']], 
                        use_container_width=True)

# ============================================================================
# SALES ANALYTICS PAGE
# ============================================================================

elif page == "📊 Sales Analytics":
    st.header("Sales Analytics")
    
    # Time period selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", 
                                   value=analyzer.df['InvoiceDate'].min())
    with col2:
        end_date = st.date_input("End Date", 
                                 value=analyzer.df['InvoiceDate'].max())
    
    # Filter data
    mask = (analyzer.df['InvoiceDate'].dt.date >= start_date) & \
           (analyzer.df['InvoiceDate'].dt.date <= end_date)
    filtered_df = analyzer.df[mask]
    
    # Sales metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        period_revenue = filtered_df['Revenue'].sum()
        st.metric("Period Revenue", f"${period_revenue:,.0f}")
    
    with col2:
        period_orders = filtered_df['InvoiceNo'].nunique()
        st.metric("Period Orders", f"{period_orders:,}")
    
    with col3:
        period_customers = filtered_df['CustomerID'].nunique()
        st.metric("Active Customers", f"{period_customers:,}")
    
    # Sales visualizations
    tab1, tab2, tab3 = st.tabs(["Temporal", "Geographic", "Product"])
    
    with tab1:
        # Hourly pattern
        hourly_sales = filtered_df.groupby('Hour')['Revenue'].mean()
        fig = px.bar(x=hourly_sales.index, y=hourly_sales.values,
                    title='Average Revenue by Hour',
                    labels={'x': 'Hour of Day', 'y': 'Avg Revenue ($)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly pattern
        filtered_df['Weekday'] = filtered_df['InvoiceDate'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']
        weekly_sales = filtered_df.groupby('Weekday')['Revenue'].sum().reindex(weekday_order)
        
        fig = px.bar(x=weekly_sales.index, y=weekly_sales.values,
                    title='Revenue by Day of Week',
                    labels={'x': 'Day', 'y': 'Revenue ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Geographic sales
        country_stats = filtered_df.groupby('Country').agg({
            'Revenue': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique'
        }).round(2)
        country_stats.columns = ['Revenue', 'Orders', 'Customers']
        country_stats = country_stats.sort_values('Revenue', ascending=False).head(20)
        
        fig = px.treemap(country_stats.reset_index(), 
                        path=['Country'], values='Revenue',
                        title='Revenue by Country (Treemap)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Top products
        product_sales = filtered_df.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(20)
        
        fig = px.bar(y=product_sales.index[::-1], x=product_sales.values[::-1],
                    orientation='h', title='Top 20 Products by Revenue')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ANOMALY DETECTION PAGE
# ============================================================================

elif page == "🔍 Anomaly Detection":
    st.header("Anomaly Detection")
    
    contamination = st.slider("Contamination Factor", 0.001, 0.05, 0.01, 0.001,
                             help="Expected proportion of anomalies")
    
    if st.button("Detect Anomalies", type="primary"):
        with st.spinner("Detecting anomalies..."):
            anomalies = analyzer.detect_anomalies(contamination=contamination)
            st.success(f"✅ Detected {len(anomalies)} anomalous transactions!")
    
    if 'Anomaly' in analyzer.df.columns:
        # Anomaly statistics
        col1, col2, col3 = st.columns(3)
        
        anomaly_df = analyzer.df[analyzer.df['Anomaly'] == -1]
        
        with col1:
            st.metric("Anomalous Transactions", f"{len(anomaly_df):,}")
        with col2:
            st.metric("Suspicious Revenue", f"${anomaly_df['Revenue'].sum():,.0f}")
        with col3:
            st.metric("Detection Rate", f"{100*len(anomaly_df)/len(analyzer.df):.2f}%")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot
            normal = analyzer.df[analyzer.df['Anomaly'] == 1].sample(min(1000, len(analyzer.df[analyzer.df['Anomaly'] == 1])))
            anomalous = analyzer.df[analyzer.df['Anomaly'] == -1]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=normal['Quantity'], y=normal['UnitPrice'],
                mode='markers', name='Normal',
                marker=dict(color='blue', size=5, opacity=0.5)
            ))
            fig.add_trace(go.Scatter(
                x=anomalous['Quantity'], y=anomalous['UnitPrice'],
                mode='markers', name='Anomaly',
                marker=dict(color='red', size=8, opacity=0.8)
            ))
            fig.update_layout(title='Anomaly Detection Results',
                            xaxis_title='Quantity', yaxis_title='Unit Price')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Anomaly score distribution
            fig = px.histogram(analyzer.df, x='Anomaly_Score',
                             title='Anomaly Score Distribution',
                             nbins=50)
            fig.add_vline(x=analyzer.df[analyzer.df['Anomaly'] == -1]['Anomaly_Score'].max(),
                         line_dash="dash", line_color="red",
                         annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        # Top anomalies
        st.subheader("🚨 Most Suspicious Transactions")
        top_anomalies = anomaly_df.nlargest(10, 'Revenue')[
            ['InvoiceDate', 'InvoiceNo', 'CustomerID', 'Quantity', 'UnitPrice', 'Revenue']
        ]
        st.dataframe(top_anomalies, use_container_width=True)

# ============================================================================
# FORECASTING PAGE
# ============================================================================

elif page == "📈 Forecasting":
    st.header("Revenue Forecasting")
    
    forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            forecast = analyzer.forecast_revenue(periods=forecast_days)
            st.success("✅ Forecast generated!")
            
            # Store forecast in session state
            st.session_state.forecast = forecast
            st.session_state.forecast_days = forecast_days
    
    if 'forecast' in st.session_state:
        # Display forecast metrics
        col1, col2, col3 = st.columns(3)
        
        forecast = st.session_state.forecast
        
        with col1:
            st.metric("Total Forecasted Revenue", 
                     f"${forecast.sum():,.0f}")
        with col2:
            st.metric("Average Daily Revenue", 
                     f"${forecast.mean():,.0f}")
        with col3:
            st.metric("Peak Day Revenue", 
                     f"${forecast.max():,.0f}")
        
        # Forecast visualization
        daily_revenue = analyzer.df.groupby(
            analyzer.df['InvoiceDate'].dt.date
        )['Revenue'].sum()
        daily_revenue.index = pd.to_datetime(daily_revenue.index)
        
        # Create forecast dates
        forecast_dates = pd.date_range(
            start=daily_revenue.index[-1] + timedelta(days=1),
            periods=len(forecast)
        )
        
        # Combined plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=daily_revenue.index[-60:],
            y=daily_revenue.values[-60:],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
            y=(forecast + 1.96 * forecast.std()).tolist() + 
              (forecast - 1.96 * forecast.std()).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence',
            showlegend=True
        ))
        
        fig.update_layout(
            title='Revenue Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MARKET ANALYSIS PAGE
# ============================================================================

elif page == "🌍 Market Analysis":
    st.header("Market Analysis")
    
    # Market statistics
    country_stats = analyzer.analyze_markets()
    
    # Top markets metrics
    top_markets = country_stats.head(5)
    cols = st.columns(len(top_markets))
    
    for idx, (country, stats) in enumerate(top_markets.iterrows()):
        with cols[idx]:
            st.metric(country[:10], 
                     f"${stats['Total_Revenue']:,.0f}",
                     f"{stats['Market_Share_%']:.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Market share pie chart
        fig = px.pie(country_stats.head(10), 
                    values='Market_Share_%',
                    names=country_stats.head(10).index,
                    title='Top 10 Markets by Share')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average order value by country
        fig = px.bar(country_stats.head(10).sort_values('Avg_Order_Value'),
                    x='Avg_Order_Value',
                    y=country_stats.head(10).sort_values('Avg_Order_Value').index,
                    orientation='h',
                    title='Average Order Value by Market')
        st.plotly_chart(fig, use_container_width=True)
    
    # Market details table
    st.subheader("Detailed Market Statistics")
    
    # Add growth potential score
    country_stats['Growth_Potential'] = (
        country_stats['Revenue_Per_Customer'] * 0.5 +
        country_stats['Avg_Order_Value'] * 0.3 +
        (100 - country_stats['Market_Share_%']) * 0.2
    ).round(2)
    
    st.dataframe(
        country_stats.style.format({
            'Total_Revenue': '${:,.0f}',
            'Avg_Order_Value': '${:,.2f}',
            'Revenue_Per_Customer': '${:,.2f}',
            'Market_Share_%': '{:.2f}%'
        }),
        use_container_width=True
    )
    
    # Market recommendations
    st.subheader("💡 Market Expansion Recommendations")
    
    high_potential = country_stats[
        (country_stats['Market_Share_%'] < 5) & 
        (country_stats['Revenue_Per_Customer'] > country_stats['Revenue_Per_Customer'].median())
    ].head(5)
    
    for country in high_potential.index:
        st.info(f"🎯 **{country}**: High customer value with only "
               f"{high_potential.loc[country, 'Market_Share_%']:.2f}% market share. "
               f"Consider targeted expansion.")

# ============================================================================
# PRODUCT ANALYSIS PAGE
# ============================================================================

elif page == "📦 Product Analysis":
    st.header("Product Analysis")
    
    with st.spinner("Analyzing products..."):
        product_stats, associations = analyzer.analyze_products(top_n=20)
    
    tabs = st.tabs(["Top Products", "Product Associations", "Category Analysis"])
    
    with tabs[0]:
        # Top products
        st.subheader("🏆 Top Products by Revenue")
        
        top_products = product_stats.head(20)
        
        fig = px.bar(top_products.reset_index(),
                    y='StockCode', x='Total_Revenue',
                    orientation='h',
                    title='Top 20 Products by Revenue',
                    hover_data=['Product'])
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Product metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Products", f"{len(product_stats):,}")
            st.metric("Top Product Revenue", 
                     f"${product_stats.iloc[0]['Total_Revenue']:,.0f}")
        
        with col2:
            st.metric("80/20 Rule", 
                     f"{(product_stats.head(int(len(product_stats)*0.2))['Total_Revenue'].sum() / product_stats['Total_Revenue'].sum() * 100):.1f}% revenue from 20% products")
            st.metric("Average Product Revenue", 
                     f"${product_stats['Total_Revenue'].mean():,.0f}")
    
    with tabs[1]:
        # Product associations
        st.subheader("🔗 Product Associations (Cross-selling)")
        
        if len(associations) > 0:
            # Filter associations
            min_lift = st.slider("Minimum Lift", 1.0, 5.0, 2.0, 0.1)
            filtered_associations = associations[associations['Lift'] >= min_lift]
            
            st.dataframe(filtered_associations.head(20), use_container_width=True)
            
            # Network graph of associations
            if len(filtered_associations) > 0:
                st.subheader("Product Association Network")
                st.info("Strong associations indicate products frequently bought together")
                
                # Create network visualization
                import networkx as nx
                
                G = nx.Graph()
                for _, row in filtered_associations.head(15).iterrows():
                    G.add_edge(row['Product_A'][:20], row['Product_B'][:20], 
                              weight=row['Lift'])
                
                pos = nx.spring_layout(G)
                
                # Create edge trace
                edge_trace = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace.append(go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=G[edge[0]][edge[1]]['weight'], color='#888'),
                        hoverinfo='none'
                    ))
                
                # Create node trace
                node_trace = go.Scatter(
                    x=[pos[node][0] for node in G.nodes()],
                    y=[pos[node][1] for node in G.nodes()],
                    mode='markers+text',
                    text=[node for node in G.nodes()],
                    textposition="top center",
                    marker=dict(size=10, color='red'),
                    hoverinfo='text'
                )
                
                fig = go.Figure(data=edge_trace + [node_trace])
                fig.update_layout(showlegend=False, height=500,
                                title='Product Association Network')
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        # Category analysis
        st.subheader("📊 Product Category Performance")
        
        # Simple categorization based on price
        product_stats['Price_Category'] = pd.cut(
            analyzer.df.groupby('StockCode')['UnitPrice'].mean(),
            bins=[0, 2, 5, 10, 1000],
            labels=['Budget', 'Standard', 'Premium', 'Luxury']
        )
        
        category_performance = product_stats.groupby('Price_Category').agg({
            'Total_Revenue': 'sum',
            'Total_Quantity': 'sum',
            'Order_Count': 'sum'
        })
        
        fig = px.bar(category_performance.reset_index(),
                    x='Price_Category', y='Total_Revenue',
                    title='Revenue by Price Category')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DECISION TREES PAGE
# ============================================================================

elif page == "🌳 Decision Trees":
    st.header("Decision Tree Analysis")
    
    # Initialize decision tree analyzer
    dt_analyzer = EnhancedDecisionTreeAnalyzer(analyzer.df)
    
    if st.button("Build Decision Tree Models", type="primary"):
        with st.spinner("Building models..."):
            models, results = dt_analyzer.build_models()
            st.success("✅ Models built successfully!")
            
            # Store in session state
            st.session_state.dt_models = models
            st.session_state.dt_results = results
    
    if 'dt_results' in st.session_state:
        # Model comparison
        st.subheader("Model Performance Comparison")
        
        results_df = pd.DataFrame(st.session_state.dt_results).T
        results_df = results_df.round(3)
        
        fig = px.bar(results_df.reset_index(), 
                     x='index', 
                     y=['accuracy', 'precision', 'recall', 'f1'],
                     title='Model Performance Metrics',
                     barmode='group')
        fig.update_layout(xaxis_title='Model', yaxis_title='Score')
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = max(st.session_state.dt_results, 
                        key=lambda x: st.session_state.dt_results[x]['f1'])
        
        st.success(f"🏆 Best Model: **{best_model.upper()}** "
                  f"(F1-Score: {st.session_state.dt_results[best_model]['f1']:.3f})")
        
        # Feature importance
        st.subheader("Feature Importance")
        
        if hasattr(st.session_state.dt_models[best_model], 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': dt_analyzer.X.columns,
                'Importance': st.session_state.dt_models[best_model].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h', title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# REPORTS PAGE
# ============================================================================

elif page == "📋 Reports":
    st.header("Business Intelligence Reports")
    
    # Generate comprehensive report
    if st.button("Generate Full Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            # Perform all analyses
            analyzer.perform_rfm_analysis()
            analyzer.perform_clustering()
            analyzer.predict_churn()
            analyzer.predict_clv()
            analyzer.detect_anomalies()
            
            st.success("✅ Report generated!")
    
    # Executive Summary
    st.subheader("📊 Executive Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Key Metrics
        - **Total Revenue**: ${:,.0f}
        - **Customer Base**: {:,} customers
        - **Average Order Value**: ${:.2f}
        - **Market Concentration**: {:.1f}% from UK
        """.format(
            analyzer.df['Revenue'].sum(),
            analyzer.df['CustomerID'].nunique(),
            analyzer.df.groupby('InvoiceNo')['Revenue'].sum().mean(),
            analyzer.df[analyzer.df['Country'] == 'United Kingdom']['Revenue'].sum() / 
            analyzer.df['Revenue'].sum() * 100
        ))
    
    with col2:
        if analyzer.customer_features is not None and 'Churn_Probability' in analyzer.customer_features.columns:
            at_risk = (analyzer.customer_features['Churn_Probability'] > 0.7).sum()
            at_risk_revenue = analyzer.customer_features[
                analyzer.customer_features['Churn_Probability'] > 0.7
            ]['Monetary'].sum()
            
            st.markdown("""
            ### Risk Assessment
            - **At-Risk Customers**: {}
            - **Revenue at Risk**: ${:,.0f}
            - **Churn Rate**: {:.1f}%
            - **Anomaly Detection**: {:.2f}% suspicious transactions
            """.format(
                at_risk,
                at_risk_revenue,
                analyzer.customer_features['Churn'].mean() * 100,
                (analyzer.df['Anomaly'] == -1).mean() * 100 if 'Anomaly' in analyzer.df.columns else 0
            ))
    
    # Downloadable reports
    st.subheader("📥 Downloadable Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if analyzer.customer_features is not None:
            csv = analyzer.customer_features.to_csv(index=False)
            st.download_button(
                label="Download Customer Segments",
                data=csv,
                file_name=f"customer_segments_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if 'Anomaly' in analyzer.df.columns:
            anomalies = analyzer.df[analyzer.df['Anomaly'] == -1]
            csv = anomalies.to_csv(index=False)
            st.download_button(
                label="Download Anomalies",
                data=csv,
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        market_stats = analyzer.analyze_markets()
        csv = market_stats.to_csv()
        st.download_button(
            label="Download Market Analysis",
            data=csv,
            file_name=f"market_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Recommendations
    st.subheader("💡 Strategic Recommendations")
    
    recommendations = []
    
    if analyzer.customer_features is not None:
        if 'Churn_Probability' in analyzer.customer_features.columns:
            high_risk_count = (analyzer.customer_features['Churn_Probability'] > 0.7).sum()
            if high_risk_count > 100:
                recommendations.append(
                    f"🎯 **Customer Retention**: {high_risk_count} customers at high risk of churn. "
                    "Implement targeted retention campaigns."
                )
        
        champions = analyzer.customer_features[
            analyzer.customer_features['Segment'] == 'Champions'
        ]
        if len(champions) > 0:
            recommendations.append(
                f"⭐ **VIP Program**: {len(champions)} champion customers generate significant revenue. "
                "Consider exclusive benefits program."
            )
    
    uk_share = analyzer.df[analyzer.df['Country'] == 'United Kingdom']['Revenue'].sum() / analyzer.df['Revenue'].sum()
    if uk_share > 0.7:
        recommendations.append(
            "🌍 **Geographic Diversification**: Over 70% revenue from UK. "
            "Explore expansion opportunities in other markets."
        )
    
    if 'Anomaly' in analyzer.df.columns:
        anomaly_rate = (analyzer.df['Anomaly'] == -1).mean()
        if anomaly_rate > 0.01:
            recommendations.append(
                f"🔍 **Fraud Prevention**: {anomaly_rate*100:.2f}% transactions flagged as suspicious. "
                "Review fraud detection protocols."
            )
    
    for rec in recommendations:
        st.info(rec)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        E-Commerce Analytics Platform v1.0 | Created by Adam Lubinsky | 
        <a href='https://github.com/AdamLubinsky'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
