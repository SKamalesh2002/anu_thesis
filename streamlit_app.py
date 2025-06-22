import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp, chi2_contingency
from scipy.stats import kruskal, ranksums, wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Medical Data Analysis Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Medical Data Analysis Dashboard")
st.markdown("### Multiple Statistical Tests Analysis for Clinical Outcomes")

# File upload section
st.sidebar.title("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file", 
    type=['csv'],
    help="Upload the medical data CSV file to begin analysis"
)

# Load data
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Try to load default file if it exists
        try:
            df = pd.read_csv("_Thesis - Sheet1.csv")
        except FileNotFoundError:
            return None
    
    # Clean INITIAL LACTATE column
    df['INITIAL LACTATE (clean)'] = df['INITIAL LACTATE'].str.extract(r'([\d.]+)').astype(float)
    
    # Clean LACTATE CLEARANCE column (matching your approach)
    df['LACTATE CLEARANCE (clean)'] = df['LACTATE CLEARANCE'].str.extract(r'([\d.]+)').astype(float)
    
    # Clean REPEAT LACTATE column
    df['REPEAT LACTATE (clean)'] = df['REPEAT LACTATE'].str.extract(r'([\d.]+)').astype(float)
    
    return df

df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ Please upload a CSV file to begin analysis")
    st.info("📋 Expected CSV columns: INITIAL LACTATE, LACTATE CLEARANCE, REPEAT LACTATE, CLINICAL OUTCOMES, AGE, K/C/O, SBP, DBP, SPO2%, CBG, HR")
    
    # Option to use sample data
    if st.sidebar.button("🎲 Generate Sample Data for Demo"):
        @st.cache_data
        def create_sample_data():
            np.random.seed(42)
            n_samples = 100
            
            sample_data = {
                'INITIAL LACTATE': [f"{np.random.normal(3.5, 1.2):.1f}" for _ in range(n_samples)],
                'LACTATE CLEARANCE': [f"{np.random.normal(25, 15):.1f}%" for _ in range(n_samples)],
                'REPEAT LACTATE': [f"{np.random.normal(2.8, 1.0):.1f}" for _ in range(n_samples)],
                'CLINICAL OUTCOMES': np.random.choice(['ALIVE', 'DEAD'], n_samples, p=[0.7, 0.3]),
                'AGE': np.random.randint(18, 90, n_samples),
                'K/C/O': np.random.choice(['CAD', 'SHTN', 'T2DM', 'CAD+SHTN', 'SHTN+T2DM', 'None'], n_samples),
                'SBP': [f"{np.random.randint(90, 180)}" for _ in range(n_samples)],
                'DBP': [f"{np.random.randint(60, 110)}" for _ in range(n_samples)],
                'SPO2%': [f"{np.random.randint(85, 100)}%" for _ in range(n_samples)],
                'CBG': [f"{np.random.randint(70, 200)}" for _ in range(n_samples)],
                'HR': [f"{np.random.randint(40, 120)}" for _ in range(n_samples)]
            }
            
            df = pd.DataFrame(sample_data)
            
            # Clean columns
            df['INITIAL LACTATE (clean)'] = df['INITIAL LACTATE'].str.extract(r'([\d.]+)').astype(float)
            df['LACTATE CLEARANCE (clean)'] = df['LACTATE CLEARANCE'].str.extract(r'([\d.]+)').astype(float)
            df['REPEAT LACTATE (clean)'] = df['REPEAT LACTATE'].str.extract(r'([\d.]+)').astype(float)
            
            return df
        
        df = create_sample_data()
        st.success("✅ Sample data generated! You can now explore the dashboard.")
    else:
        st.stop()

try:
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        ["Overview", "Initial Lactate Analysis", "Lactate Clearance Analysis", "Repeat Lactate Analysis", "Age Analysis", "CAD Analysis", "SHTN+T2DM Analysis", "Unstable Hemodynamic Analysis", "Combined Analysis"]
    )
    
    if analysis_type == "Overview":
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            alive_count = len(df[df['CLINICAL OUTCOMES'] == 'ALIVE'])
            st.metric("Alive", alive_count)
        with col3:
            dead_count = len(df[df['CLINICAL OUTCOMES'] == 'DEAD'])
            st.metric("Dead", dead_count)
        with col4:
            survival_rate = (alive_count / len(df)) * 100
            st.metric("Survival Rate", f"{survival_rate:.1f}%")
        
        # Outcome distribution pie chart
        fig_pie = px.pie(
            values=[alive_count, dead_count],
            names=['ALIVE', 'DEAD'],
            title="Clinical Outcomes Distribution",
            color_discrete_map={'ALIVE': '#2E8B57', 'DEAD': '#DC143C'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Age distribution
        fig_age = px.histogram(
            df, x='AGE', color='CLINICAL OUTCOMES',
            title="Age Distribution by Clinical Outcome",
            nbins=20,
            color_discrete_map={'ALIVE': '#2E8B57', 'DEAD': '#DC143C'}
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    elif analysis_type == "Initial Lactate Analysis":
        st.header("🧪 Initial Lactate vs Clinical Outcomes")
        
        # Filter data
        filtered_df = df[['INITIAL LACTATE (clean)', 'CLINICAL OUTCOMES']].dropna()
        alive_group = filtered_df[filtered_df['CLINICAL OUTCOMES'] == 'ALIVE']['INITIAL LACTATE (clean)']
        dead_group = filtered_df[filtered_df['CLINICAL OUTCOMES'] == 'DEAD']['INITIAL LACTATE (clean)']
        
        # Mann-Whitney U Test
        u_stat, p_value = mannwhitneyu(alive_group, dead_group, alternative='two-sided')
        
        # Display test results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("U-Statistic", f"{u_stat:.2f}")
        with col2:
            st.metric("P-Value", f"{p_value:.4f}")
        with col3:
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            st.metric("Result", significance)
        
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=alive_group, name='ALIVE', marker_color='#2E8B57'))
        fig_box.add_trace(go.Box(y=dead_group, name='DEAD', marker_color='#DC143C'))
        fig_box.update_layout(
            title="Initial Lactate Distribution by Clinical Outcome",
            yaxis_title="Initial Lactate (mmol/L)",
            xaxis_title="Clinical Outcome"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Violin plot
        fig_violin = go.Figure()
        fig_violin.add_trace(go.Violin(y=alive_group, name='ALIVE', box_visible=True, 
                                     line_color='#2E8B57', fillcolor='rgba(46,139,87,0.5)'))
        fig_violin.add_trace(go.Violin(y=dead_group, name='DEAD', box_visible=True,
                                     line_color='#DC143C', fillcolor='rgba(220,20,60,0.5)'))
        fig_violin.update_layout(
            title="Initial Lactate Distribution (Violin Plot)",
            yaxis_title="Initial Lactate (mmol/L)"
        )
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ALIVE Group**")
            st.write(f"Mean: {alive_group.mean():.2f} mmol/L")
            st.write(f"Median: {alive_group.median():.2f} mmol/L")
            st.write(f"Std Dev: {alive_group.std():.2f} mmol/L")
            st.write(f"Count: {len(alive_group)}")
        
        with col2:
            st.write("**DEAD Group**")
            st.write(f"Mean: {dead_group.mean():.2f} mmol/L")
            st.write(f"Median: {dead_group.median():.2f} mmol/L")
            st.write(f"Std Dev: {dead_group.std():.2f} mmol/L")
            st.write(f"Count: {len(dead_group)}")
    
    elif analysis_type == "Lactate Clearance Analysis":
        st.header("🔄 Lactate Clearance vs Clinical Outcomes")
        
        # Filter data (matching your approach)
        filtered_df = df[['LACTATE CLEARANCE (clean)', 'CLINICAL OUTCOMES']].dropna()
        alive_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'ALIVE']['LACTATE CLEARANCE (clean)']
        dead_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'DEAD']['LACTATE CLEARANCE (clean)']
        
        # Multiple Statistical Tests
        st.subheader("📊 Statistical Test Results")
        
        # 1. Mann-Whitney U Test
        u_stat, p_mw = mannwhitneyu(alive_group, dead_group, alternative='two-sided')
        
        # 2. Welch's t-test (unequal variances)
        t_stat, p_ttest = ttest_ind(alive_group, dead_group, equal_var=False)
        
        # 3. Kolmogorov-Smirnov test
        ks_stat, p_ks = ks_2samp(alive_group, dead_group)
        
        # 4. Permutation test
        from scipy.stats import permutation_test
        def stat_func(x, y):
            return np.mean(x) - np.mean(y)
        perm_test = permutation_test((alive_group, dead_group), stat_func, n_resamples=10000)
        p_perm = perm_test.pvalue
        
        # 5. Bootstrap test
        from scipy.stats import bootstrap
        boot_test = bootstrap((alive_group, dead_group), stat_func, n_resamples=10000)
        p_boot = boot_test.confidence_interval[0]
        
        # Create test results table
        test_results = {
            'Test': ['T-test', 'Mann-Whitney U', 'Kolmogorov-Smirnov', 'Permutation', 'Bootstrap'],
            'Statistic': [t_stat, u_stat, ks_stat, perm_test.statistic, boot_test.confidence_interval[0]],
            'P-Value': [p_ttest, p_mw, p_ks, p_perm, p_boot],
            'Significant': ['Yes' if p < 0.05 else 'No' for p in [p_ttest, p_mw, p_ks, p_perm, p_boot]]
        }
        
        results_df = pd.DataFrame(test_results)
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Highlight most significant test
        p_values = [p_ttest, p_mw, p_ks, p_perm, p_boot]
        min_p = min(p_values)
        best_test_idx = p_values.index(min_p)
        best_test = test_results['Test'][best_test_idx]
        st.success(f"🎯 Most significant result: **{best_test}** (p = {min_p:.4f})")
        
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=alive_group, name='ALIVE', marker_color='#2E8B57'))
        fig_box.add_trace(go.Box(y=dead_group, name='DEAD', marker_color='#DC143C'))
        fig_box.update_layout(
            title="Lactate Clearance Distribution by Clinical Outcome",
            yaxis_title="Lactate Clearance (%)",
            xaxis_title="Clinical Outcome"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=alive_group, name='ALIVE', opacity=0.7, 
                                      marker_color='#2E8B57', nbinsx=20))
        fig_hist.add_trace(go.Histogram(x=dead_group, name='DEAD', opacity=0.7,
                                      marker_color='#DC143C', nbinsx=20))
        fig_hist.update_layout(
            title="Lactate Clearance Distribution Histogram",
            xaxis_title="Lactate Clearance (%)",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ALIVE Group**")
            st.write(f"Mean: {alive_group.mean():.2f}%")
            st.write(f"Median: {alive_group.median():.2f}%")
            st.write(f"Std Dev: {alive_group.std():.2f}%")
            st.write(f"Count: {len(alive_group)}")
        
        with col2:
            st.write("**DEAD Group**")
            st.write(f"Mean: {dead_group.mean():.2f}%")
            st.write(f"Median: {dead_group.median():.2f}%")
            st.write(f"Std Dev: {dead_group.std():.2f}%")
            st.write(f"Count: {len(dead_group)}")
    
    elif analysis_type == "Repeat Lactate Analysis":
        st.header("🔁 Repeat Lactate vs Clinical Outcomes")
        
        # Filter data
        filtered_df = df[['REPEAT LACTATE (clean)', 'CLINICAL OUTCOMES']].dropna()
        alive_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'ALIVE']['REPEAT LACTATE (clean)']
        dead_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'DEAD']['REPEAT LACTATE (clean)']
        
        # Multiple Statistical Tests
        st.subheader("📊 Statistical Test Results")
        
        # 1. Mann-Whitney U Test
        u_stat, p_mw = mannwhitneyu(alive_group, dead_group, alternative='two-sided')
        
        # 2. Welch's t-test (unequal variances)
        t_stat, p_ttest = ttest_ind(alive_group, dead_group, equal_var=False)
        
        # 3. Kolmogorov-Smirnov test
        ks_stat, p_ks = ks_2samp(alive_group, dead_group)
        
        # 4. Permutation test
        from scipy.stats import permutation_test
        def stat_func(x, y):
            return np.mean(x) - np.mean(y)
        perm_test = permutation_test((alive_group, dead_group), stat_func, n_resamples=10000)
        p_perm = perm_test.pvalue
        
        # 5. Bootstrap test
        from scipy.stats import bootstrap
        boot_test = bootstrap((alive_group, dead_group), stat_func, n_resamples=10000)
        p_boot = boot_test.confidence_interval[0]
        
        # Create test results table
        test_results = {
            'Test': ['T-test', 'Mann-Whitney U', 'Kolmogorov-Smirnov', 'Permutation', 'Bootstrap'],
            'Statistic': [t_stat, u_stat, ks_stat, perm_test.statistic, boot_test.confidence_interval[0]],
            'P-Value': [p_ttest, p_mw, p_ks, p_perm, p_boot],
            'Significant': ['Yes' if p < 0.05 else 'No' for p in [p_ttest, p_mw, p_ks, p_perm, p_boot]]
        }
        
        results_df = pd.DataFrame(test_results)
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Highlight most significant test
        p_values = [p_ttest, p_mw, p_ks, p_perm, p_boot]
        min_p = min(p_values)
        best_test_idx = p_values.index(min_p)
        best_test = test_results['Test'][best_test_idx]
        st.success(f"🎯 Most significant result: **{best_test}** (p = {min_p:.4f})")
        
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=alive_group, name='ALIVE', marker_color='#2E8B57'))
        fig_box.add_trace(go.Box(y=dead_group, name='DEAD', marker_color='#DC143C'))
        fig_box.update_layout(
            title="Repeat Lactate Distribution by Clinical Outcome",
            yaxis_title="Repeat Lactate (mmol/L)",
            xaxis_title="Clinical Outcome"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=alive_group, name='ALIVE', opacity=0.7, 
                                      marker_color='#2E8B57', nbinsx=20))
        fig_hist.add_trace(go.Histogram(x=dead_group, name='DEAD', opacity=0.7,
                                      marker_color='#DC143C', nbinsx=20))
        fig_hist.update_layout(
            title="Repeat Lactate Distribution Histogram",
            xaxis_title="Repeat Lactate (mmol/L)",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ALIVE Group**")
            st.write(f"Mean: {alive_group.mean():.2f} mmol/L")
            st.write(f"Median: {alive_group.median():.2f} mmol/L")
            st.write(f"Std Dev: {alive_group.std():.2f} mmol/L")
            st.write(f"Count: {len(alive_group)}")
        
        with col2:
            st.write("**DEAD Group**")
            st.write(f"Mean: {dead_group.mean():.2f} mmol/L")
            st.write(f"Median: {dead_group.median():.2f} mmol/L")
            st.write(f"Std Dev: {dead_group.std():.2f} mmol/L")
            st.write(f"Count: {len(dead_group)}")
    
    elif analysis_type == "Age Analysis":
        st.header("👥 Age vs Clinical Outcomes")
        
        # Filter data
        filtered_df = df[['AGE', 'CLINICAL OUTCOMES']].dropna()
        alive_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'ALIVE']['AGE']
        dead_group = filtered_df[filtered_df['CLINICAL OUTCOMES'].str.upper() == 'DEAD']['AGE']
        
        # Multiple Statistical Tests
        st.subheader("📊 Statistical Test Results")
        
        # 1. Mann-Whitney U Test
        u_stat, p_mw = mannwhitneyu(alive_group, dead_group, alternative='two-sided')
        
        # 2. Welch's t-test (unequal variances)
        t_stat, p_ttest = ttest_ind(alive_group, dead_group, equal_var=False)
        
        # 3. Kolmogorov-Smirnov test
        ks_stat, p_ks = ks_2samp(alive_group, dead_group)
        
        # 4. Permutation test
        from scipy.stats import permutation_test
        def stat_func(x, y):
            return np.mean(x) - np.mean(y)
        perm_test = permutation_test((alive_group, dead_group), stat_func, n_resamples=10000)
        p_perm = perm_test.pvalue
        
        # 5. Bootstrap test
        from scipy.stats import bootstrap
        boot_test = bootstrap((alive_group, dead_group), stat_func, n_resamples=10000)
        p_boot = boot_test.confidence_interval[0]
        
        # Create test results table
        test_results = {
            'Test': ['T-test', 'Mann-Whitney U', 'Kolmogorov-Smirnov', 'Permutation', 'Bootstrap'],
            'Statistic': [t_stat, u_stat, ks_stat, perm_test.statistic, boot_test.confidence_interval[0]],
            'P-Value': [p_ttest, p_mw, p_ks, p_perm, p_boot],
            'Significant': ['Yes' if p < 0.05 else 'No' for p in [p_ttest, p_mw, p_ks, p_perm, p_boot]]
        }
        
        results_df = pd.DataFrame(test_results)
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Highlight most significant test
        p_values = [p_ttest, p_mw, p_ks, p_perm, p_boot]
        min_p = min(p_values)
        best_test_idx = p_values.index(min_p)
        best_test = test_results['Test'][best_test_idx]
        st.success(f"🎯 Most significant result: **{best_test}** (p = {min_p:.4f})")
        
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=alive_group, name='ALIVE', marker_color='#2E8B57'))
        fig_box.add_trace(go.Box(y=dead_group, name='DEAD', marker_color='#DC143C'))
        fig_box.update_layout(
            title="Age Distribution by Clinical Outcome",
            yaxis_title="Age (years)",
            xaxis_title="Clinical Outcome"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=alive_group, name='ALIVE', opacity=0.7, 
                                      marker_color='#2E8B57', nbinsx=20))
        fig_hist.add_trace(go.Histogram(x=dead_group, name='DEAD', opacity=0.7,
                                      marker_color='#DC143C', nbinsx=20))
        fig_hist.update_layout(
            title="Age Distribution Histogram",
            xaxis_title="Age (years)",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**ALIVE Group**")
            st.write(f"Mean: {alive_group.mean():.1f} years")
            st.write(f"Median: {alive_group.median():.1f} years")
            st.write(f"Std Dev: {alive_group.std():.1f} years")
            st.write(f"Count: {len(alive_group)}")
        
        with col2:
            st.write("**DEAD Group**")
            st.write(f"Mean: {dead_group.mean():.1f} years")
            st.write(f"Median: {dead_group.median():.1f} years")
            st.write(f"Std Dev: {dead_group.std():.1f} years")
            st.write(f"Count: {len(dead_group)}")
    
    elif analysis_type == "CAD Analysis":
        st.header("❤️ CAD vs Clinical Outcomes")
        
        # Filter data for CAD analysis
        filtered_df = df[['K/C/O', 'CLINICAL OUTCOMES']].dropna()
        
        # Check if CAD is present in the K/C/O string
        filtered_df['has_CAD'] = filtered_df['K/C/O'].str.contains('CAD', case=False, na=False)
        
        cad_group = filtered_df[filtered_df['has_CAD'] == True]['CLINICAL OUTCOMES']
        no_cad_group = filtered_df[filtered_df['has_CAD'] == False]['CLINICAL OUTCOMES']
        
        # Create contingency table
        cad_alive = len(cad_group[cad_group.str.upper() == 'ALIVE'])
        cad_dead = len(cad_group[cad_group.str.upper() == 'DEAD'])
        no_cad_alive = len(no_cad_group[no_cad_group.str.upper() == 'ALIVE'])
        no_cad_dead = len(no_cad_group[no_cad_group.str.upper() == 'DEAD'])
        
        contingency_table = [[cad_alive, cad_dead], [no_cad_alive, no_cad_dead]]
        
        # Use Fisher's exact test if any cell has count < 5, otherwise chi-square
        if min(cad_alive, cad_dead, no_cad_alive, no_cad_dead) < 5:
            from scipy.stats import fisher_exact
            odds_ratio, p_chi2 = fisher_exact(contingency_table)
            chi2_stat = odds_ratio
            test_name = "Fisher's Exact Test"
        else:
            chi2_stat, p_chi2, dof, expected = chi2_contingency(contingency_table)
            test_name = "Chi-square Test"
        
        # Display test results
        st.subheader("📊 Statistical Test Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{test_name} Statistic", f"{chi2_stat:.4f}")
        with col2:
            st.metric("P-Value", f"{p_chi2:.4f}")
        with col3:
            significance = "Significant" if p_chi2 < 0.05 else "Not Significant"
            st.metric("Result", significance)
        
        # Contingency table display
        st.subheader("📋 Contingency Table")
        contingency_df = pd.DataFrame({
            'CAD': [cad_alive, cad_dead],
            'No CAD': [no_cad_alive, no_cad_dead]
        }, index=['ALIVE', 'DEAD'])
        st.dataframe(contingency_df, use_container_width=True)
        
        # Stacked bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='ALIVE', x=['CAD', 'No CAD'], y=[cad_alive, no_cad_alive], marker_color='#2E8B57'))
        fig_bar.add_trace(go.Bar(name='DEAD', x=['CAD', 'No CAD'], y=[cad_dead, no_cad_dead], marker_color='#DC143C'))
        fig_bar.update_layout(
            title="Clinical Outcomes by CAD Status",
            xaxis_title="CAD Status",
            yaxis_title="Count",
            barmode='stack'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Survival rates
        st.subheader("📈 Survival Rates")
        col1, col2 = st.columns(2)
        
        with col1:
            cad_total = cad_alive + cad_dead
            cad_survival_rate = (cad_alive / cad_total * 100) if cad_total > 0 else 0
            st.write("**CAD Patients**")
            st.write(f"Total: {cad_total}")
            st.write(f"Alive: {cad_alive}")
            st.write(f"Dead: {cad_dead}")
            st.write(f"Survival Rate: {cad_survival_rate:.1f}%")
        
        with col2:
            no_cad_total = no_cad_alive + no_cad_dead
            no_cad_survival_rate = (no_cad_alive / no_cad_total * 100) if no_cad_total > 0 else 0
            st.write("**No CAD Patients**")
            st.write(f"Total: {no_cad_total}")
            st.write(f"Alive: {no_cad_alive}")
            st.write(f"Dead: {no_cad_dead}")
            st.write(f"Survival Rate: {no_cad_survival_rate:.1f}%")
    
    elif analysis_type == "SHTN+T2DM Analysis":
        st.header("💔 SHTN+T2DM vs Clinical Outcomes")
        
        # Filter data for SHTN+T2DM analysis
        filtered_df = df[['K/C/O', 'CLINICAL OUTCOMES']].dropna()
        
        # Check if both SHTN and T2DM are present in the K/C/O string
        filtered_df['has_SHTN_T2DM'] = (filtered_df['K/C/O'].str.contains('SHTN', case=False, na=False) | 
                                        filtered_df['K/C/O'].str.contains('T2DM', case=False, na=False))
        
        shtn_t2dm_group = filtered_df[filtered_df['has_SHTN_T2DM'] == True]['CLINICAL OUTCOMES']
        no_shtn_t2dm_group = filtered_df[filtered_df['has_SHTN_T2DM'] == False]['CLINICAL OUTCOMES']
        
        # Create contingency table
        shtn_t2dm_alive = len(shtn_t2dm_group[shtn_t2dm_group.str.upper() == 'ALIVE'])
        shtn_t2dm_dead = len(shtn_t2dm_group[shtn_t2dm_group.str.upper() == 'DEAD'])
        no_shtn_t2dm_alive = len(no_shtn_t2dm_group[no_shtn_t2dm_group.str.upper() == 'ALIVE'])
        no_shtn_t2dm_dead = len(no_shtn_t2dm_group[no_shtn_t2dm_group.str.upper() == 'DEAD'])
        
        contingency_table = [[shtn_t2dm_alive, shtn_t2dm_dead], [no_shtn_t2dm_alive, no_shtn_t2dm_dead]]
        
        # Use Fisher's exact test if any cell has count < 5, otherwise chi-square
        if min(shtn_t2dm_alive, shtn_t2dm_dead, no_shtn_t2dm_alive, no_shtn_t2dm_dead) < 5:
            from scipy.stats import fisher_exact
            odds_ratio, p_chi2 = fisher_exact(contingency_table)
            chi2_stat = odds_ratio
            test_name = "Fisher's Exact Test"
        else:
            chi2_stat, p_chi2, dof, expected = chi2_contingency(contingency_table)
            test_name = "Chi-square Test"
        
        # Display test results
        st.subheader("📊 Statistical Test Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{test_name} Statistic", f"{chi2_stat:.4f}")
        with col2:
            st.metric("P-Value", f"{p_chi2:.4f}")
        with col3:
            significance = "Significant" if p_chi2 < 0.05 else "Not Significant"
            st.metric("Result", significance)
        
        # Contingency table display
        st.subheader("📋 Contingency Table")
        contingency_df = pd.DataFrame({
            'SHTN+T2DM': [shtn_t2dm_alive, shtn_t2dm_dead],
            'Others': [no_shtn_t2dm_alive, no_shtn_t2dm_dead]
        }, index=['ALIVE', 'DEAD'])
        st.dataframe(contingency_df, use_container_width=True)
        
        # Stacked bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='ALIVE', x=['SHTN+T2DM', 'Others'], y=[shtn_t2dm_alive, no_shtn_t2dm_alive], marker_color='#2E8B57'))
        fig_bar.add_trace(go.Bar(name='DEAD', x=['SHTN+T2DM', 'Others'], y=[shtn_t2dm_dead, no_shtn_t2dm_dead], marker_color='#DC143C'))
        fig_bar.update_layout(
            title="Clinical Outcomes by SHTN+T2DM Status",
            xaxis_title="Patient Group",
            yaxis_title="Count",
            barmode='stack'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Survival rates
        st.subheader("📈 Survival Rates")
        col1, col2 = st.columns(2)
        
        with col1:
            shtn_t2dm_total = shtn_t2dm_alive + shtn_t2dm_dead
            shtn_t2dm_survival_rate = (shtn_t2dm_alive / shtn_t2dm_total * 100) if shtn_t2dm_total > 0 else 0
            st.write("**SHTN+T2DM Patients**")
            st.write(f"Total: {shtn_t2dm_total}")
            st.write(f"Alive: {shtn_t2dm_alive}")
            st.write(f"Dead: {shtn_t2dm_dead}")
            st.write(f"Survival Rate: {shtn_t2dm_survival_rate:.1f}%")
        
        with col2:
            no_shtn_t2dm_total = no_shtn_t2dm_alive + no_shtn_t2dm_dead
            no_shtn_t2dm_survival_rate = (no_shtn_t2dm_alive / no_shtn_t2dm_total * 100) if no_shtn_t2dm_total > 0 else 0
            st.write("**Other Patients**")
            st.write(f"Total: {no_shtn_t2dm_total}")
            st.write(f"Alive: {no_shtn_t2dm_alive}")
            st.write(f"Dead: {no_shtn_t2dm_dead}")
            st.write(f"Survival Rate: {no_shtn_t2dm_survival_rate:.1f}%")
    
    elif analysis_type == "Unstable Hemodynamic Analysis":
        st.header("⚠️ Unstable Hemodynamic vs Clinical Outcomes")
        
        # Clean vital signs data
        df['SBP_clean'] = df['SBP'].str.extract(r'(\d+)').astype(float)
        df['DBP_clean'] = df['DBP'].str.extract(r'(\d+)').astype(float)
        df['SPO2_clean'] = df['SPO2%'].str.extract(r'(\d+)').astype(float)
        df['CBG_clean'] = df['CBG'].str.extract(r'(\d+)').astype(float)
        df['HR_clean'] = df['HR'].str.extract(r'(\d+)').astype(float)
        
        # Filter data for hemodynamic analysis
        filtered_df = df[['SBP_clean', 'DBP_clean', 'SPO2_clean', 'CBG_clean', 'HR_clean', 'CLINICAL OUTCOMES']].dropna()
        
        # Define unstable hemodynamics criteria
        filtered_df['unstable_hemo'] = (
            (filtered_df['SBP_clean'] < 120) |
            (filtered_df['DBP_clean'] < 80) |
            (filtered_df['SPO2_clean'] < 90) |
            (filtered_df['CBG_clean'] < 75) |
            (filtered_df['HR_clean'] < 45)
        )
        
        unstable_group = filtered_df[filtered_df['unstable_hemo'] == True]['CLINICAL OUTCOMES']
        stable_group = filtered_df[filtered_df['unstable_hemo'] == False]['CLINICAL OUTCOMES']
        
        # Create contingency table
        unstable_alive = len(unstable_group[unstable_group.str.upper() == 'ALIVE'])
        unstable_dead = len(unstable_group[unstable_group.str.upper() == 'DEAD'])
        stable_alive = len(stable_group[stable_group.str.upper() == 'ALIVE'])
        stable_dead = len(stable_group[stable_group.str.upper() == 'DEAD'])
        
        contingency_table = [[unstable_alive, unstable_dead], [stable_alive, stable_dead]]
        
        # Use Fisher's exact test if any cell has count < 5, otherwise chi-square
        if min(unstable_alive, unstable_dead, stable_alive, stable_dead) < 5:
            from scipy.stats import fisher_exact
            odds_ratio, p_chi2 = fisher_exact(contingency_table)
            chi2_stat = odds_ratio
            test_name = "Fisher's Exact Test"
        else:
            chi2_stat, p_chi2, dof, expected = chi2_contingency(contingency_table)
            test_name = "Chi-square Test"
        
        # Display test results
        st.subheader("📊 Statistical Test Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{test_name} Statistic", f"{chi2_stat:.4f}")
        with col2:
            st.metric("P-Value", f"{p_chi2:.4f}")
        with col3:
            significance = "Significant" if p_chi2 < 0.05 else "Not Significant"
            st.metric("Result", significance)
        
        # Criteria display
        st.subheader("🩺 Unstable Hemodynamic Criteria")
        st.write("**Patients classified as unstable if ANY of the following:**")
        st.write("• SBP < 120 mmHg")
        st.write("• DBP < 80 mmHg")
        st.write("• SPO2 < 90%")
        st.write("• CBG < 75 mg/dL")
        st.write("• HR < 45 BPM")
        
        # Contingency table display
        st.subheader("📋 Contingency Table")
        contingency_df = pd.DataFrame({
            'Unstable Hemodynamics': [unstable_alive, unstable_dead],
            'Stable Hemodynamics': [stable_alive, stable_dead]
        }, index=['ALIVE', 'DEAD'])
        st.dataframe(contingency_df, use_container_width=True)
        
        # Stacked bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='ALIVE', x=['Unstable', 'Stable'], y=[unstable_alive, stable_alive], marker_color='#2E8B57'))
        fig_bar.add_trace(go.Bar(name='DEAD', x=['Unstable', 'Stable'], y=[unstable_dead, stable_dead], marker_color='#DC143C'))
        fig_bar.update_layout(
            title="Clinical Outcomes by Hemodynamic Status",
            xaxis_title="Hemodynamic Status",
            yaxis_title="Count",
            barmode='stack'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Survival rates
        st.subheader("📈 Survival Rates")
        col1, col2 = st.columns(2)
        
        with col1:
            unstable_total = unstable_alive + unstable_dead
            unstable_survival_rate = (unstable_alive / unstable_total * 100) if unstable_total > 0 else 0
            st.write("**Unstable Hemodynamics**")
            st.write(f"Total: {unstable_total}")
            st.write(f"Alive: {unstable_alive}")
            st.write(f"Dead: {unstable_dead}")
            st.write(f"Survival Rate: {unstable_survival_rate:.1f}%")
        
        with col2:
            stable_total = stable_alive + stable_dead
            stable_survival_rate = (stable_alive / stable_total * 100) if stable_total > 0 else 0
            st.write("**Stable Hemodynamics**")
            st.write(f"Total: {stable_total}")
            st.write(f"Alive: {stable_alive}")
            st.write(f"Dead: {stable_dead}")
            st.write(f"Survival Rate: {stable_survival_rate:.1f}%")
    
    elif analysis_type == "Combined Analysis":
        st.header("🔬 Combined Analysis Dashboard")
        
        # Prepare data for all tests
        initial_df = df[['INITIAL LACTATE (clean)', 'CLINICAL OUTCOMES']].dropna()
        clearance_df = df[['LACTATE CLEARANCE (clean)', 'CLINICAL OUTCOMES']].dropna()
        repeat_df = df[['REPEAT LACTATE (clean)', 'CLINICAL OUTCOMES']].dropna()
        age_df = df[['AGE', 'CLINICAL OUTCOMES']].dropna()
        cad_df = df[['K/C/O', 'CLINICAL OUTCOMES']].dropna()
        shtn_t2dm_df = df[['K/C/O', 'CLINICAL OUTCOMES']].dropna()
        
        # Clean vital signs for hemodynamic analysis
        df['SBP_clean'] = df['SBP'].str.extract(r'(\d+)').astype(float)
        df['DBP_clean'] = df['DBP'].str.extract(r'(\d+)').astype(float)
        df['SPO2_clean'] = df['SPO2%'].str.extract(r'(\d+)').astype(float)
        df['CBG_clean'] = df['CBG'].str.extract(r'(\d+)').astype(float)
        df['HR_clean'] = df['HR'].str.extract(r'(\d+)').astype(float)
        hemo_df = df[['SBP_clean', 'DBP_clean', 'SPO2_clean', 'CBG_clean', 'HR_clean', 'CLINICAL OUTCOMES']].dropna()
        
        # Initial Lactate groups
        initial_alive = initial_df[initial_df['CLINICAL OUTCOMES'] == 'ALIVE']['INITIAL LACTATE (clean)']
        initial_dead = initial_df[initial_df['CLINICAL OUTCOMES'] == 'DEAD']['INITIAL LACTATE (clean)']
        
        # Clearance groups
        clearance_alive = clearance_df[clearance_df['CLINICAL OUTCOMES'] == 'ALIVE']['LACTATE CLEARANCE (clean)']
        clearance_dead = clearance_df[clearance_df['CLINICAL OUTCOMES'] == 'DEAD']['LACTATE CLEARANCE (clean)']
        
        # Repeat lactate groups
        repeat_alive = repeat_df[repeat_df['CLINICAL OUTCOMES'] == 'ALIVE']['REPEAT LACTATE (clean)']
        repeat_dead = repeat_df[repeat_df['CLINICAL OUTCOMES'] == 'DEAD']['REPEAT LACTATE (clean)']
        
        # Age groups
        age_alive = age_df[age_df['CLINICAL OUTCOMES'] == 'ALIVE']['AGE']
        age_dead = age_df[age_df['CLINICAL OUTCOMES'] == 'DEAD']['AGE']
        
        # Perform tests
        u_stat_initial, p_val_initial = mannwhitneyu(initial_alive, initial_dead, alternative='two-sided')
        u_stat_clearance, p_val_clearance = mannwhitneyu(clearance_alive, clearance_dead, alternative='two-sided')
        u_stat_repeat, p_val_repeat = mannwhitneyu(repeat_alive, repeat_dead, alternative='two-sided')
        u_stat_age, p_val_age = mannwhitneyu(age_alive, age_dead, alternative='two-sided')
        
        # CAD analysis
        cad_df['has_CAD'] = cad_df['K/C/O'].str.contains('CAD', case=False, na=False)
        cad_group = cad_df[cad_df['has_CAD'] == True]['CLINICAL OUTCOMES']
        no_cad_group = cad_df[cad_df['has_CAD'] == False]['CLINICAL OUTCOMES']
        cad_alive_count = len(cad_group[cad_group.str.upper() == 'ALIVE'])
        cad_dead_count = len(cad_group[cad_group.str.upper() == 'DEAD'])
        no_cad_alive_count = len(no_cad_group[no_cad_group.str.upper() == 'ALIVE'])
        no_cad_dead_count = len(no_cad_group[no_cad_group.str.upper() == 'DEAD'])
        contingency = [[cad_alive_count, cad_dead_count], [no_cad_alive_count, no_cad_dead_count]]
        
        if min(cad_alive_count, cad_dead_count, no_cad_alive_count, no_cad_dead_count) < 5:
            from scipy.stats import fisher_exact
            chi2_stat_cad, p_val_cad = fisher_exact(contingency)
        else:
            chi2_stat_cad, p_val_cad, _, _ = chi2_contingency(contingency)
        
        # SHTN+T2DM analysis
        shtn_t2dm_df['has_SHTN_T2DM'] = (shtn_t2dm_df['K/C/O'].str.contains('SHTN', case=False, na=False) & 
                                         shtn_t2dm_df['K/C/O'].str.contains('T2DM', case=False, na=False))
        shtn_t2dm_group = shtn_t2dm_df[shtn_t2dm_df['has_SHTN_T2DM'] == True]['CLINICAL OUTCOMES']
        no_shtn_t2dm_group = shtn_t2dm_df[shtn_t2dm_df['has_SHTN_T2DM'] == False]['CLINICAL OUTCOMES']
        shtn_t2dm_alive_count = len(shtn_t2dm_group[shtn_t2dm_group.str.upper() == 'ALIVE'])
        shtn_t2dm_dead_count = len(shtn_t2dm_group[shtn_t2dm_group.str.upper() == 'DEAD'])
        no_shtn_t2dm_alive_count = len(no_shtn_t2dm_group[no_shtn_t2dm_group.str.upper() == 'ALIVE'])
        no_shtn_t2dm_dead_count = len(no_shtn_t2dm_group[no_shtn_t2dm_group.str.upper() == 'DEAD'])
        contingency_shtn_t2dm = [[shtn_t2dm_alive_count, shtn_t2dm_dead_count], [no_shtn_t2dm_alive_count, no_shtn_t2dm_dead_count]]
        
        if min(shtn_t2dm_alive_count, shtn_t2dm_dead_count, no_shtn_t2dm_alive_count, no_shtn_t2dm_dead_count) < 5:
            from scipy.stats import fisher_exact
            chi2_stat_shtn_t2dm, p_val_shtn_t2dm = fisher_exact(contingency_shtn_t2dm)
        else:
            chi2_stat_shtn_t2dm, p_val_shtn_t2dm, _, _ = chi2_contingency(contingency_shtn_t2dm)
        
        # Hemodynamic analysis
        hemo_df['unstable_hemo'] = (
            (hemo_df['SBP_clean'] < 120) |
            (hemo_df['DBP_clean'] < 80) |
            (hemo_df['SPO2_clean'] < 90) |
            (hemo_df['CBG_clean'] < 75) |
            (hemo_df['HR_clean'] < 45)
        )
        unstable_hemo_group = hemo_df[hemo_df['unstable_hemo'] == True]['CLINICAL OUTCOMES']
        stable_hemo_group = hemo_df[hemo_df['unstable_hemo'] == False]['CLINICAL OUTCOMES']
        unstable_hemo_alive = len(unstable_hemo_group[unstable_hemo_group.str.upper() == 'ALIVE'])
        unstable_hemo_dead = len(unstable_hemo_group[unstable_hemo_group.str.upper() == 'DEAD'])
        stable_hemo_alive = len(stable_hemo_group[stable_hemo_group.str.upper() == 'ALIVE'])
        stable_hemo_dead = len(stable_hemo_group[stable_hemo_group.str.upper() == 'DEAD'])
        contingency_hemo = [[unstable_hemo_alive, unstable_hemo_dead], [stable_hemo_alive, stable_hemo_dead]]
        
        if min(unstable_hemo_alive, unstable_hemo_dead, stable_hemo_alive, stable_hemo_dead) < 5:
            from scipy.stats import fisher_exact
            chi2_stat_hemo, p_val_hemo = fisher_exact(contingency_hemo)
        else:
            chi2_stat_hemo, p_val_hemo, _, _ = chi2_contingency(contingency_hemo)
        
        # Test results table
        st.subheader("🧪 Statistical Test Results")
        results_data = {
            'Test': ['Initial Lactate vs Outcomes', 'Lactate Clearance vs Outcomes', 'Repeat Lactate vs Outcomes', 'Age vs Outcomes', 'CAD vs Outcomes', 'SHTN+T2DM vs Outcomes', 'Unstable Hemodynamics vs Outcomes'],
            'Test Type': ['Mann-Whitney U', 'Mann-Whitney U', 'Mann-Whitney U', 'Mann-Whitney U', 'Fisher/Chi-square', 'Fisher/Chi-square', 'Fisher/Chi-square'],
            'Statistic': [u_stat_initial, u_stat_clearance, u_stat_repeat, u_stat_age, chi2_stat_cad, chi2_stat_shtn_t2dm, chi2_stat_hemo],
            'P-Value': [p_val_initial, p_val_clearance, p_val_repeat, p_val_age, p_val_cad, p_val_shtn_t2dm, p_val_hemo],
            'Significant (α=0.05)': [
                'Yes' if p_val_initial < 0.05 else 'No',
                'Yes' if p_val_clearance < 0.05 else 'No',
                'Yes' if p_val_repeat < 0.05 else 'No',
                'Yes' if p_val_age < 0.05 else 'No',
                'Yes' if p_val_cad < 0.05 else 'No',
                'Yes' if p_val_shtn_t2dm < 0.05 else 'No',
                'Yes' if p_val_hemo < 0.05 else 'No'
            ]
        }
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Initial Lactate scatter plot
            fig_scatter1 = px.scatter(
                initial_df, 
                x='CLINICAL OUTCOMES', 
                y='INITIAL LACTATE (clean)',
                color='CLINICAL OUTCOMES',
                title="Initial Lactate by Outcome",
                color_discrete_map={'ALIVE': '#2E8B57', 'DEAD': '#DC143C'}
            )
            fig_scatter1.add_hline(y=initial_df['INITIAL LACTATE (clean)'].mean(), 
                                 line_dash="dash", line_color="gray",
                                 annotation_text=f"Mean: {initial_df['INITIAL LACTATE (clean)'].mean():.2f}")
            st.plotly_chart(fig_scatter1, use_container_width=True)
        
        with col2:
            # Lactate Clearance scatter plot
            fig_scatter2 = px.scatter(
                clearance_df, 
                x='CLINICAL OUTCOMES', 
                y='LACTATE CLEARANCE (clean)',
                color='CLINICAL OUTCOMES',
                title="Lactate Clearance by Outcome",
                color_discrete_map={'ALIVE': '#2E8B57', 'DEAD': '#DC143C'}
            )
            fig_scatter2.add_hline(y=clearance_df['LACTATE CLEARANCE (clean)'].mean(), 
                                 line_dash="dash", line_color="gray",
                                 annotation_text=f"Mean: {clearance_df['LACTATE CLEARANCE (clean)'].mean():.2f}")
            st.plotly_chart(fig_scatter2, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        correlation_df = df[['INITIAL LACTATE (clean)', 'LACTATE CLEARANCE (clean)']].dropna()
        correlation = correlation_df.corr().iloc[0, 1]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Correlation Coefficient", f"{correlation:.3f}")
        
        with col2:
            fig_corr = px.scatter(
                correlation_df,
                x='INITIAL LACTATE (clean)',
                y='LACTATE CLEARANCE (clean)',
                title="Initial Lactate vs Lactate Clearance"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

except FileNotFoundError:
    st.error("❌ CSV file not found. Please upload your data file using the sidebar.")
    st.info("💡 Tip: Use the file uploader in the sidebar or generate sample data to test the dashboard.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.info("💡 Please check your data format and try again.")

# Footer
st.markdown("---")
st.markdown("*Dashboard created for medical data analysis using multiple statistical tests*")