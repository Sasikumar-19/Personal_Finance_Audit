import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from multi_loader import load_file, load_single_csv, unlock_excel

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Personal Finance Audit",
    page_icon  = "💰",
    layout     = "wide",
)

CAT_COLORS = {
    'Food & Dining'  : '#E74C3C', 'Transport'   : '#3498DB',
    'Shopping'       : '#9B59B6', 'Bills & Rent': '#E67E22',
    'Entertainment'  : '#1ABC9C', 'Health'      : '#27AE60',
    'Education'      : '#2980B9', 'Travel'      : '#F39C12',
    'Others'         : '#95A5A6', 'UPI Transfer': '#EC407A',
}

# ── Analysis helpers ──────────────────────────────────────────────────────────
def run_anomaly_detection(df):
    le = LabelEncoder()
    df = df.copy()
    df['Category_Code'] = le.fit_transform(df['Category'])
    df['Day_of_Week']   = df['Date'].dt.dayofweek
    df['Amount_Rank']   = df.groupby('Category')['Amount'].rank(pct=True)
    X = df[['Amount','Month_Num','Category_Code','Day_of_Week','Amount_Rank']]
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df['Anomaly']       = model.fit_predict(X)
    df['Anomaly_Score'] = model.decision_function(X)
    df['Anomaly_Label'] = df['Anomaly'].map({1:'Normal', -1:'🚨 Anomaly'})
    return df

def run_clustering(df):
    monthly = (df.groupby(['Month','Month_Num','Year','Category'])['Amount']
                 .sum().unstack(fill_value=0).reset_index())
    monthly = monthly.sort_values(['Year','Month_Num']).reset_index(drop=True)
    cat_cols = [c for c in monthly.columns if c not in ['Month','Month_Num','Year']]
    monthly['Total_Spend'] = monthly[cat_cols].sum(axis=1)
    if len(monthly) < 3:
        monthly['Persona'] = '🔵 Balanced'
        return monthly, cat_cols, None, {}, None
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(monthly[cat_cols])
    kmeans   = KMeans(n_clusters=3, random_state=42, n_init=10)
    monthly['Cluster'] = kmeans.fit_predict(X_scaled)
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_), columns=cat_cols)
    centroids['Total'] = centroids[cat_cols].sum(axis=1)
    rank = centroids.sort_values('Total').index.tolist()
    PERSONA_MAP = {rank[0]:'🟢 Saver', rank[1]:'🔵 Balanced', rank[2]:'🔴 High Spender'}
    monthly['Persona'] = monthly['Cluster'].map(PERSONA_MAP)
    return monthly, cat_cols, centroids, PERSONA_MAP, scaler

def generate_recommendations(df, monthly):
    recs = []
    cat_totals = df.groupby('Category')['Amount'].sum()
    top_cat = cat_totals.idxmax()
    top_pct = cat_totals.max() / cat_totals.sum() * 100
    if top_pct > 25:
        recs.append(('🔴 High', top_cat,
            f'{top_cat} is {top_pct:.1f}% of total spend. '
            f'A 10% cut saves ₹{cat_totals.max()*0.10:,.0f}/year.'))
    hs = monthly[monthly['Persona']=='🔴 High Spender']['Month'].tolist()
    if hs:
        recs.append(('🔴 High', 'Monthly Budget',
            f'{len(hs)} High Spender month(s): {", ".join(hs[:2])}. '
            f'Set alert at ₹{monthly["Total_Spend"].mean():,.0f}/month.'))
    if 'Food & Dining' in cat_totals.index:
        food_m = cat_totals['Food & Dining'] / 12
        if food_m > 2000:
            recs.append(('🟡 Medium', 'Food & Dining',
                f'₹{food_m:,.0f}/month on food. '
                f'Cooking 3x/week saves ₹{food_m*0.25:,.0f}/month.'))
    if 'Entertainment' in cat_totals.index:
        ent_m = cat_totals['Entertainment'] / 12
        if ent_m > 800:
            recs.append(('🟡 Medium', 'Subscriptions',
                f'₹{ent_m:,.0f}/month on entertainment. '
                f'Audit subscriptions — save ₹{ent_m*0.3:,.0f}/month.'))
    saver = monthly[monthly['Persona']=='🟢 Saver']
    if len(saver):
        potential = (monthly['Total_Spend'].mean() - saver['Total_Spend'].mean()) * 12
        recs.append(('🟢 Opportunity', 'Savings Potential',
            f'Matching Saver months year-round saves ₹{potential:,.0f}/year.'))
    if not recs:
        recs.append(('🟢 Good', 'Overall',
            'Your spending looks balanced! Keep tracking monthly.'))
    return recs

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💰 Finance Audit")
    st.markdown("---")
    mode = st.radio("Choose input mode",
                    ["🎯 Demo Data", "📂 Upload CSV / Excel"])

    # ── Demo mode ─────────────────────────────────────────────────────────
    if mode == "🎯 Demo Data":
        if st.button("▶ Load Demo Data", type="primary"):
            try:
                demo_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'sample_bank_statement.csv')
                df_loaded = load_single_csv(demo_path)
                st.session_state['df'] = df_loaded
                st.success(f"✅ {len(df_loaded):,} transactions loaded!")
            except Exception as e:
                st.error(f"Demo file error: {e}")

    # ── Upload mode ───────────────────────────────────────────────────────
    else:
        st.info(
            "🔒 **Privacy Notice**\n\n"
            "Your file and password are used **only for this session**. "
            "Nothing is stored after you close this tab."
        )
        uploaded = st.file_uploader(
            "Upload bank / UPI statement",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
        )
        password = st.text_input(
            "Password (if file is protected)",
            type="password",
            placeholder="Leave blank if not password protected",
        )

        if uploaded and st.button("▶ Analyse", type="primary"):
            with st.spinner("Reading your statement…"):
                try:
                    file_bytes = uploaded.read()
                    fname      = uploaded.name

                    # ── Unlock Excel if password given ─────────────────────
                    if fname.endswith(('.xlsx', '.xls')) and password:
                        unlocked, err = unlock_excel(file_bytes, password)
                        if err:
                            st.error(f"❌ {err}")
                            st.stop()
                        # unlocked is already a BytesIO — pass directly
                        unlocked.seek(0)
                        df_loaded = load_file(unlocked, filename=fname)
                    else:
                        # CSV or unprotected Excel — wrap bytes in BytesIO
                        df_loaded = load_file(
                            io.BytesIO(file_bytes), filename=fname)

                    st.session_state['df'] = df_loaded
                    st.success(f"✅ {len(df_loaded):,} transactions loaded!")

                except Exception as e:
                    st.error(
                        f"**Could not read this statement.**\n\n"
                        f"{e}\n\n"
                        f"*Try downloading your statement as CSV from netbanking.*"
                    )

    # ── Sidebar stats + filter ─────────────────────────────────────────────
    if 'df' in st.session_state:
        raw = st.session_state['df']
        st.markdown("---")
        st.markdown(f"**{len(raw):,}** transactions")
        st.markdown(f"**₹{raw['Amount'].sum():,.0f}** total spend")
        st.markdown("---")
        all_cats = sorted(raw['Category'].unique())
        selected = st.multiselect("Filter categories",
                                  options=all_cats, default=all_cats)
        st.session_state['cats'] = selected

# ── Main content ──────────────────────────────────────────────────────────────
st.title("📊 Personal Finance Anomaly & Lifestyle Audit")

if 'df' not in st.session_state:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spend",   "—")
    col2.metric("Transactions",  "—")
    col3.metric("Anomalies",     "—")
    st.info("👈 **Load demo data or upload your statement** from the sidebar to begin.")
    st.stop()

# Apply category filter
raw      = st.session_state['df']
sel_cats = st.session_state.get('cats', sorted(raw['Category'].unique()))
df       = raw[raw['Category'].isin(sel_cats)].copy() if sel_cats else raw.copy()

# Run ML
df                                              = run_anomaly_detection(df)
monthly, cat_cols, centroids, PERSONA_MAP, _   = run_clustering(df)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📈 Trends", "🚨 Anomalies", "🧬 Personas", "💡 Tips"])

# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Spend",     f"₹{df['Amount'].sum():,.0f}")
    c2.metric("Avg / Month",     f"₹{monthly['Total_Spend'].mean():,.0f}")
    c3.metric("Transactions",    f"{len(df):,}")
    c4.metric("Anomalies Found", f"{(df['Anomaly']==-1).sum()}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            df.groupby('Category')['Amount'].sum().reset_index(),
            values='Amount', names='Category', hole=0.45,
            color='Category', color_discrete_map=CAT_COLORS,
            title='Spending by Category')
        fig.update_traces(textposition='outside', textinfo='label+percent')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top10 = (df.groupby('Description')['Amount'].sum()
                   .sort_values(ascending=False).head(10).reset_index())
        fig2 = px.bar(top10.sort_values('Amount'), x='Amount',
                      y='Description', orientation='h',
                      title='Top 10 Merchants',
                      color='Amount', color_continuous_scale='Blues')
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

# ── TAB 2: TRENDS ─────────────────────────────────────────────────────────────
with tab2:
    fig3 = px.line(monthly, x='Month', y='Total_Spend',
                   title='Monthly Spending Trend', markers=True)
    avg = monthly['Total_Spend'].mean()
    fig3.add_hline(y=avg, line_dash='dash', line_color='red',
                   annotation_text=f'Avg ₹{avg:,.0f}')
    fig3.update_traces(line_color='#2E75B6', line_width=2.5, marker_size=8)
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

    m_cat = (df.groupby(['Month','Month_Num','Year','Category'])['Amount']
               .sum().reset_index().sort_values(['Year','Month_Num']))
    fig4 = px.bar(m_cat, x='Month', y='Amount', color='Category',
                  color_discrete_map=CAT_COLORS, barmode='stack',
                  title='Category Breakdown by Month')
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)

# ── TAB 3: ANOMALIES ──────────────────────────────────────────────────────────
with tab3:
    n_anom = (df['Anomaly'] == -1).sum()
    st.markdown(f"### 🚨 {n_anom} anomalous transactions detected")
    fig5 = px.scatter(df, x='Date', y='Amount', color='Anomaly_Label',
                      color_discrete_map={'Normal':'#95A5A6','🚨 Anomaly':'#E74C3C'},
                      size='Amount', size_max=30,
                      hover_data=['Description','Category','Anomaly_Score'],
                      title='Transaction Timeline — Anomalies Highlighted')
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("#### Flagged Transactions")
    anom_df = (df[df['Anomaly'] == -1]
               .sort_values('Anomaly_Score')
               [['Date','Description','Amount','Category','Anomaly_Score']]
               .copy())
    anom_df['Amount']        = anom_df['Amount'].apply(lambda x: f'₹{x:,.0f}')
    anom_df['Anomaly_Score'] = anom_df['Anomaly_Score'].round(4)
    anom_df['Date']          = pd.to_datetime(anom_df['Date']).dt.strftime('%d %b %Y')
    st.dataframe(anom_df, use_container_width=True)

# ── TAB 4: PERSONAS ───────────────────────────────────────────────────────────
with tab4:
    pcolors = {'🟢 Saver':'#27AE60','🔵 Balanced':'#2980B9','🔴 High Spender':'#E74C3C'}
    col1, col2 = st.columns(2)
    with col1:
        fig6 = px.bar(monthly, x='Month', y='Total_Spend', color='Persona',
                      color_discrete_map=pcolors,
                      title='Monthly Spending Personas', text='Persona')
        fig6.update_traces(textposition='outside')
        fig6.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig6, use_container_width=True)
    with col2:
        if centroids is not None and PERSONA_MAP:
            fig7 = go.Figure()
            for cid, persona in PERSONA_MAP.items():
                vals  = list(centroids[cat_cols].iloc[cid].values) 
                vals += [vals[0]]
                theta = cat_cols + [cat_cols[0]]
                fig7.add_trace(go.Scatterpolar(
                    r=vals, theta=theta, fill='toself', name=persona,
                    line_color=pcolors.get(persona,'grey'),
                    fillcolor=pcolors.get(persona,'grey'), opacity=0.3))
            fig7.update_layout(title='Spending Radar by Persona', height=420)
            st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("Need at least 3 months of data for persona clustering.")

# ── TAB 5: TIPS ───────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 💡 Personalised Savings Recommendations")
    for priority, area, tip in generate_recommendations(df, monthly):
        st.info(f"**{priority} | {area}**\n\n{tip}")
