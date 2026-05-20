import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================
# 1. PAGE SETUP & DATA LOADING
# ==========================================
st.set_page_config(page_title="User Analytics Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("analytics_data_may_whole.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["fs_date"] = pd.to_datetime(df["fs_date"], errors="coerce")
    df = df.sort_values(["user_id", "fs_date"])
    first_dates = df.groupby("user_id", as_index=False).agg(
        first_acquired_date=("fs_date", "min")
    )
    latest_rows = df.groupby("user_id", as_index=False).tail(1).copy()
    latest_rows = latest_rows.rename(columns={"period": "latest_period"})
    user_level = latest_rows.merge(first_dates, on="user_id", how="left")
    return user_level

df = load_data()


with st.sidebar:
    st.header("🔍 Global Filters")
    
    # 1. Signup Calendar Date Range Filter
    min_date = df['first_acquired_date'].min().date()
    max_date = df['first_acquired_date'].max().date()
    selected_date_range = st.date_input(
        "Signup Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # 2. Demographics & Channels Multiselects
    all_channels = sorted([c for c in df['acquisition_channel'].dropna().unique()])
    selected_channels = st.multiselect("Acquisition Channel", options=all_channels, default=all_channels)

    df['age_group'] = pd.cut(
    df['userage'], 
    bins=[0, 18, 25, 35, 45, 55, 120], 
    labels=['<18', '18-25', '26-35', '36-45', '46-55', '55+'])

    all_age_groups = ['<18', '18-25', '26-35', '36-45', '46-55', '55+']
    selected_ages = st.multiselect("User Age Group", options=all_age_groups, default=all_age_groups)


    df['user_type_groups'] = pd.cut(
    df['user_type'], 
    bins=[-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
    labels=['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.10'])

    all_user_type_groups = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.10']
    selected_user_types = st.multiselect("User Type Group", options=all_user_type_groups, default=all_user_type_groups)

    all_tiers = sorted([t for t in df['tier'].dropna().unique()])
    selected_tiers = st.multiselect("User Tier", options=all_tiers, default=all_tiers)

    # 3. Account Status Dropdown
    account_options = ["All", "Single Account Only", "Multi Account Only"]
    selected_acc = st.selectbox("Account Status", options=account_options, index=0)


# ==========================================
# FILTER PROCESSING ENGINE
# ==========================================
df = df.copy()

if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    df = df[(df['first_acquired_date'].dt.date >= start_date) & (df['first_acquired_date'].dt.date <= end_date)]

if selected_channels:
    df = df[df['acquisition_channel'].isin(selected_channels)]
if selected_ages:
    df = df[df['age_group'].isin(selected_ages)]
if selected_user_types:
    df = df[df['user_type_groups'].isin(selected_user_types)]
if selected_tiers:
    df = df[df['tier'].isin(selected_tiers)]

if selected_acc == "Single Account Only":
    df = df[df['is_multi_account'] == False]
elif selected_acc == "Multi Account Only":
    df = df[df['is_multi_account'] == True]

st.title("New User Spend Behavior & Retention Dashboard")
# Summary KPI Header Row (Directly below Title)
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.info(f"📊 **Filtered Cohort Size:** {len(df):,} users (out of {len(df):,})")
with m_col2:
    active_rate = df['stayed_active_180d'].mean() * 100 if len(df) > 0 else 0
    st.success(f"📈 **Cohort 180d Retention:** {active_rate:.2f}%")
with m_col3:
    total_spenders = (df['spend_usd_f7d'] > 0).sum()
    st.warning(f"💎 **Early Spenders Count:** {total_spenders:,}")

st.markdown("---")

# Data Enrichment for visualization purposes
df['signup_month'] = df['first_acquired_date'].dt.strftime('%Y-%m')
# Ensure categorical/boolean types are appropriately typed for Plotly
df['stayed_active_180d'] = df['stayed_active_180d'].astype(bool)
df['is_multi_account'] = df['is_multi_account'].astype(bool)
df['tier'] = df['tier'].astype(str)

# ==========================================
# 2. TABS SETUP
# ==========================================
tabs = st.tabs([
    "1. Acquisition", 
    "2. Early Spend", 
    "3. Retention & Activity", 
    "4. Spend-Retention",
    "5. Engagement Depth",
])

# ==========================================
# TAB 1: NEW USER ACQUISITION
# ==========================================
with tabs[0]:
    st.header("New User Acquisition")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("1. Month wise new users count")
        monthly_users = df.groupby('signup_month').size().reset_index(name='count').dropna()
        st.plotly_chart(px.line(monthly_users, x='signup_month', y='count', markers=True), use_container_width=True,key="key07")
        
        st.subheader("3. Channels bringing highest-value new users (High value = retained 180 days %)")
        high_value_counts = df[df['stayed_active_180d'] == True].groupby('acquisition_channel').size().reset_index(name='high_value_users')
        total_users_counts = df.groupby('acquisition_channel').size().reset_index(name='total_users')
        high_value = pd.merge(total_users_counts, high_value_counts, on='acquisition_channel', how='left').fillna(0)
        high_value["high_value_users%"] = (high_value["high_value_users"] / high_value["total_users"]) * 100
        st.plotly_chart(px.bar(high_value, x='acquisition_channel', y='high_value_users%', color='acquisition_channel', labels={'high_value_users%': 'High Value Users (%)'}), use_container_width=True,key="key06")

        st.subheader("5. Channels producing best retained users (Considering 180 days retention)")
        retention_by_channel = df.groupby('acquisition_channel')['stayed_active_180d'].mean().reset_index(name='retention_rate')
        st.plotly_chart(px.bar(retention_by_channel, x='acquisition_channel', y='retention_rate', color='acquisition_channel'), use_container_width=True,key="key05")
        
        st.subheader("7. Channels with highest multi-account share")
        multi_acc = df.groupby('acquisition_channel')['is_multi_account'].mean().reset_index(name='multi_acc_rate')
        st.plotly_chart(px.bar(multi_acc, x='acquisition_channel', y='multi_acc_rate', color='acquisition_channel'), use_container_width=True,key="key04")

    with c2:
        st.subheader("2. Volume of new users by acquisition channel")
        channel_vol = df['acquisition_channel'].value_counts().reset_index()
        channel_vol.columns = ['acquisition_channel', 'count']
        st.plotly_chart(px.pie(channel_vol, names='acquisition_channel', values='count'), use_container_width=True,key="key03")

        st.subheader("4. Percentile distribution of 180d revenue per user")
        # Generate percentiles from 0.1 to 0.99
        percentile_values = [round(x, 2) for x in np.arange(0.1, 1.0, 0.01)]
        rev_percentiles = df['total_revenue_180d'].quantile(percentile_values).reset_index()
        rev_percentiles.columns = ['percentile', '180d_revenue']
        
        fig_percentile = px.line(
            rev_percentiles, 
            x='percentile', 
            y='180d_revenue', 
            markers=True,
            labels={'percentile': 'Percentile Value', '180d_revenue': '180D Revenue per User'}
        )
        st.plotly_chart(fig_percentile, use_container_width=True,key="key02")

        st.subheader("6. Channels bringing early spenders who don't retain")
        df['is_churned_spender'] = (df['spend_usd_f7d'] > 0) & (df['spend_usd_f7d'] > df['fs_amount']) & (~df['stayed_active_180d'])
        churn_by_chan = df.groupby('acquisition_channel')['is_churned_spender'].mean().reset_index(name='churned_spenders%')
        churn_by_chan['churned_spenders%'] *= 100
        st.plotly_chart(px.bar(churn_by_chan, x='acquisition_channel', y='churned_spenders%', color='acquisition_channel', labels={'churned_spenders%': 'Churned Spenders (% of Total Users)'}), use_container_width=True,key="key01")

        st.subheader("8. Acquisition channel mix across tiers")
        mix = df.groupby(['tier', 'acquisition_channel']).size().reset_index(name='count')
        fig_mix = px.bar(
            mix, 
            x='tier', 
            y='count', 
            color='acquisition_channel', 
            barmode='stack', 
            labels={'count': 'Percentage (%)'}
        )
        # Apply 100% stack layout fix here
        fig_mix.update_layout(barnorm='percent')
        st.plotly_chart(fig_mix, use_container_width=True)

# ==========================================
# TAB 2: EARLY SPEND BEHAVIOR
# ==========================================
with tabs[1]:
    st.header("Early Spend Behavior")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("1. Average first spend amount")
        st.metric("Avg First Spend (Amount)", f"${df['fs_amount'].mean():.2f}")
        #st.plotly_chart(px.histogram(df[df['fs_amount'] > 0], x='fs_amount', nbins=50), use_container_width=True)
        
        st.subheader("3. Proportion generating zero spend in first 7 days")
        zero_spend_prop = (df['spend_usd_f7d'] == 0).mean() * 100
        st.plotly_chart(px.pie(names=['Zero Spend (F7D)', 'Spent'], values=[zero_spend_prop, 100-zero_spend_prop]), use_container_width=True,key="key0")
        
        st.subheader("5. High-value spend segments (180d active)")
        st.plotly_chart(px.scatter(df, x='spend_usd_f7d', y='total_revenue_180d', color='stayed_active_180d'), use_container_width=True,key="key1")

    with c2:
        st.subheader("2. Median first spend amount")
        st.metric("Median First Spend", f"${df['fs_amount'].median():.2f}")
        #st.plotly_chart(px.box(df[df['fs_amount'] > 0], y='fs_amount'), use_container_width=True)

        st.subheader("4. Distribution of spend in first 7 days")
        st.plotly_chart(px.histogram(df[df['spend_usd_f7d'] > 0], x='spend_usd_f7d', nbins=1000, log_y=True), use_container_width=True)

        st.subheader("6. High-value spend segments (180d active) - Excluding the whale spender")
        filtered_df = df[df['total_revenue_180d'] < 200000]
        st.plotly_chart(px.scatter(filtered_df, x='spend_usd_f7d', y='total_revenue_180d', color='stayed_active_180d'), use_container_width=True,key="key2")

# ==========================================
# TAB 3: RETENTION AND ACTIVITY
# ==========================================
with tabs[2]:
    st.header("Retention & Activity")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("1. Users active for 180 days")
        retention_pct = df['stayed_active_180d'].mean() * 100
        st.metric("180d Retention Rate", f"{retention_pct:.2f}%")
        
        st.subheader("3. Retention vs Logins (First 3 & 7 days)")
        login_ret = df.groupby('stayed_active_180d')[['logins_f3d', 'logins_f7d']].mean().reset_index()
        login_ret['stayed_active_180d'] = login_ret['stayed_active_180d'].astype(str)
        st.plotly_chart(px.bar(login_ret.melt(id_vars='stayed_active_180d'), x='variable', y='value', color='stayed_active_180d', barmode='group'), use_container_width=True)
        
        st.subheader("5. Retention by Acquisition Channel")
        ret_chan = df.groupby('acquisition_channel')['stayed_active_180d'].mean().reset_index()
        st.plotly_chart(px.bar(ret_chan, x='acquisition_channel', y='stayed_active_180d'), use_container_width=True)

        st.subheader("7. Retention by Tier")
        ret_tier = df.groupby('tier')['stayed_active_180d'].mean().reset_index()
        st.plotly_chart(px.bar(ret_tier, x='tier', y='stayed_active_180d'), use_container_width=True,key="key3")

    with c2:
        st.subheader("2. Early activity vs Long-term retention")
        login_ret_7d = df.groupby('logins_f7d')['stayed_active_180d'].mean().reset_index(name='retention_rate')
        login_ret_7d['retention_rate'] *= 100
        login_ret_7d['text_label'] = login_ret_7d['retention_rate'].map(lambda x: f"{x:.1f}%")
        login_ret_7d = login_ret_7d[login_ret_7d['logins_f7d'] > 0]
        fig_login_ret = px.bar(
            login_ret_7d, 
            x='logins_f7d', 
            y='retention_rate',
            text='text_label',
            labels={'logins_f7d': 'Number of Logged-in Days (F7D)', 'retention_rate': '180d Retention Rate (%)'}
        )
        fig_login_ret.update_traces(textposition='outside')
        st.plotly_chart(fig_login_ret, use_container_width=True,key="key4")
        
        st.subheader("4. Retention vs First-week game activity")
        bins = [-1, 100, 200, 500, 1000, 1500, float('inf')]
        labels = ['0-100', '100-200', '200-500', '500-1000', '1000-1500', '1500+']
        df['games_f7d_bin'] = pd.cut(df['games_f7d'], bins=bins, labels=labels)
        game_ret_7d = df.groupby('games_f7d_bin', observed=False)['stayed_active_180d'].agg(['mean', 'count']).reset_index()
        game_ret_7d = game_ret_7d.rename(columns={'mean': 'retention_rate', 'count': 'user_count'})
        game_ret_7d['retention_rate'] *= 100
        game_ret_7d['text_label'] = game_ret_7d['retention_rate'].map(lambda x: f"{x:.1f}%")
        fig_game_ret = px.bar(
            game_ret_7d, 
            x='games_f7d_bin', 
            y='retention_rate',
            text='text_label',
            hover_data={'user_count': ':,', 'retention_rate': ':.1f%'},
            labels={
                'games_f7d_bin': 'Number of Games Played (F7D)', 
                'retention_rate': '180d Retention Rate (%)',
                'user_count': 'Total Users'
            }
        )
        fig_game_ret.update_traces(textposition='outside')
        st.plotly_chart(fig_game_ret, use_container_width=True,key="key5")
        
        
# ==========================================
# TAB 4: SPEND-RETENTION LINKAGE
# ==========================================
with tabs[3]:
    st.header("Spend-Retention Linkage")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("1. Retention by First Spend Amount")
        bins = [-1,10,20,30,40,50,100,150, 200, 400, 600, 800, 1000, 1500, float('inf')]
        labels = ['0-10','10-20','20-30', '30-40','40-50','50-100', '100-150', '150-200', '200-400', '400-600', '600-800', '800-1000', '1000-1500', '1500+']
        df['fs_amount_bin'] = pd.cut(df['fs_amount'], bins=bins, labels=labels)
        
        fs_ret = df.groupby('fs_amount_bin', observed=False)['stayed_active_180d'].agg(['mean', 'count']).reset_index()
        fs_ret = fs_ret.rename(columns={'mean': 'retention_rate', 'count': 'user_count'})
        fs_ret['retention_rate'] *= 100
        fs_ret['text_label'] = fs_ret['retention_rate'].map(lambda x: f"{x:.1f}%")
        fig_fs_ret = px.bar(
            fs_ret, 
            x='fs_amount_bin', 
            y='retention_rate',
            text='text_label',
            hover_data={'user_count': ':,', 'retention_rate': ':.1f%'},
            labels={
                'fs_amount_bin': 'First Spend Amount ($)', 
                'retention_rate': '180d Retention Rate (%)',
                'user_count': 'Total Users'
            }
        )
        fig_fs_ret.update_traces(textposition='outside')
        st.plotly_chart(fig_fs_ret, use_container_width=True,key="key6")

        

        st.subheader("3. Spend threshold for material retention improvement")
        df['spend_f7d_bin'] = pd.qcut(df['spend_usd_f7d'].rank(method='first'), q=10, labels=False)
        thresh = df.groupby('spend_f7d_bin')['stayed_active_180d'].mean().reset_index()
        st.plotly_chart(px.line(thresh, x='spend_f7d_bin', y='stayed_active_180d', markers=True, title="Deciles of 7D Spend vs Retention Rate"), use_container_width=True,key="key60")

        

    with c2:
        st.subheader("2. Retention by First 7 days Spend Amount")
        bins = [-1,10,20,30,40,50,100,150, 200, 400, 600, 800, 1000, 1500, float('inf')]
        labels = ['0-10','10-20','20-30', '30-40','40-50','50-100', '100-150', '150-200', '200-400', '400-600', '600-800', '800-1000', '1000-1500', '1500+']
        df['spend_usd_f7d_bin'] = pd.cut(df['spend_usd_f7d'], bins=bins, labels=labels)
        
        fs_ret = df.groupby('spend_usd_f7d_bin', observed=False)['stayed_active_180d'].agg(['mean', 'count']).reset_index()
        fs_ret = fs_ret.rename(columns={'mean': 'retention_rate', 'count': 'user_count'})
        fs_ret['retention_rate'] *= 100
        fs_ret['text_label'] = fs_ret['retention_rate'].map(lambda x: f"{x:.1f}%")
        fig_fs_ret = px.bar(
            fs_ret, 
            x='spend_usd_f7d_bin', 
            y='retention_rate',
            text='text_label',
            hover_data={'user_count': ':,', 'retention_rate': ':.1f%'},
            labels={
                'spend_usd_f7d_bin': 'First 7 Days Spend Amount ($)', 
                'retention_rate': '180d Retention Rate (%)',
                'user_count': 'Total Users'
            }
        )
        fig_fs_ret.update_traces(textposition='outside')
        st.plotly_chart(fig_fs_ret, use_container_width=True,key="key7")



# ==========================================
# TAB 5: ENGAGEMENT DEPTH
# ==========================================
with tabs[4]:
    st.header("Engagement Depth")
    st.subheader("Early Engagement Metrics Correlation Matrix")

    # 1. Define the metrics to test for correlation
    corr_metrics = ['logins_f7d', 'games_f7d', 'revenue_f7d', 'spend_cnt_f7d', 'spend_usd_f7d','stayed_active_180d']
    
    # 2. Compute the Pearson correlation matrix
    corr_matrix = df[corr_metrics].corr()
    
    # 3. Build the annotated heatmap
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',  # Automatically adds the correlation numbers inside the squares
        aspect="auto",
        color_continuous_scale='RdBu_r',  # Red-to-Blue scale (classic for correlations)
        zmin=-1,                          # Force scale bounds from -1 to 1
        zmax=1,
        labels=dict(x="Metrics", y="Metrics", color="Correlation")
    )
    
    # 4. Tweak layout details for maximum readability
    fig_corr.update_layout(
        margin=dict(l=40, r=40, t=20, b=20),
        height=450
    )
    
    st.plotly_chart(fig_corr, use_container_width=True,key="engagement_depth")
