import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SPENDIGO", page_icon="💸", layout="wide")

st.markdown("""
<style>
.section-title{font-size:18px;font-weight:600;color:#1a1a1a;margin:24px 0 10px;
  padding-bottom:6px;border-bottom:2px solid #7F77DD}
.bq-box{background:#f0f4ff;border-radius:10px;padding:14px 18px;margin-bottom:10px;
  border-left:4px solid #7F77DD}
.ab-result-box{background:#f0f4ff;border-radius:10px;padding:14px 18px;
  margin-bottom:10px;border:1px solid #d0d8ff}
.ab-sig{color:#0F6E56;font-weight:600}
.ab-nosig{color:#A32D2D;font-weight:600}
.info-box{background:#fff8e1;border-radius:10px;padding:12px 16px;
  border-left:4px solid #FAC775;margin-bottom:10px;font-size:14px}
</style>
""", unsafe_allow_html=True)


# ── LOAD & FEATURE ENGINEERING ───────────────────────────────
@st.cache_data
def load_data():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, 'genz_financial_dataset_v3.csv'))
    df['Date']         = pd.to_datetime(df['Date'])
    df['Year']         = df['Date'].dt.year
    df['Month']        = df['Date'].dt.month
    df['Day']          = df['Date'].dt.day
    df['day_of_week']  = df['Date'].dt.weekday
    df['is_weekday']   = (df['day_of_week'] < 5).astype(int)
    df['is_ramadan']   = df['Month'].isin([3, 4]).astype(int)
    df['is_harbolnas'] = (
        ((df['Month'] == 11) & (df['Day'] == 11)) |
        ((df['Month'] == 12) & (df['Day'] == 12))
    ).astype(int)
    df['amount_log'] = np.log1p(df['Amount'])
    df['is_income']  = (df['Type'] == 'Income').astype(int)
    return df

df = load_data()
df_f = df.copy()
df_exp = df_f[df_f['Type'] == 'Expense'].copy()
df_inc = df_f[df_f['Type'] == 'Income'].copy()
expense_only = df[df['Type'] == 'Expense'].copy()

# ── HEADER ────────────────────────────────────────────────────
st.title("💸 SPENDIGO — Analisis Keuangan Mahasiswa Gen-Z")
st.caption(
    f"Dataset: {len(df_f):,} transaksi | {df_f['User_ID'].nunique()} user | "
    f"{df_f['Date'].min().date()} s/d {df_f['Date'].max().date()}"
)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Problem & Data",
    "📊 EDA",
    "🧪 A/B Testing",
    "🤖 Model RF",
    "📈 LSTM",
    "📚 Data Dictionary"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — PROBLEM DISCOVERY & DATA
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Problem Discovery</div>', unsafe_allow_html=True)
    st.markdown("""
    **Latar Belakang:** Mahasiswa Gen-Z sering mengalami kesulitan dalam mengelola keuangan pribadi
    akibat minimnya kesadaran mencatat pengeluaran dan kurangnya tools yang mudah digunakan.
    """)

    st.markdown('<div class="section-title">Business Questions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="bq-box">
    <b>BQ1:</b> Bagaimana merancang sistem pencatatan keuangan yang mudah digunakan sehingga
    pengguna mau konsisten mencatat transaksi?<br>
    <i>→ Dianalisis melalui pola frekuensi transaksi per user type, kategori yang paling sering
    dicatat, dan konsistensi pencatatan bulanan antar kepribadian finansial.</i>
    </div>
    <div class="bq-box">
    <b>BQ2:</b> Bagaimana menyajikan data keuangan dalam bentuk yang sederhana namun informatif?<br>
    <i>→ Dianalisis melalui visualisasi distribusi pengeluaran per kategori, tren bulanan,
    perbandingan income vs expense, dan pola musiman (Ramadan, Harbolnas).</i>
    </div>
    <div class="bq-box">
    <b>BQ3:</b> Bagaimana analisis data sederhana dapat memberikan insight yang relevan bagi pengguna?<br>
    <i>→ Dianalisis melalui A/B Testing (Mann-Whitney & Chi-Square), feature importance model,
    dan prediksi pengeluaran bulanan berbasis LSTM untuk rekomendasi personal.</i>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Struktur Data</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Baris",    f"{len(df):,}")
    col2.metric("Jumlah Kolom",   f"{len(df.columns)}")
    col3.metric("Jumlah User",    f"{df['User_ID'].nunique()}")
    col4.metric("Missing Values", f"{df.isnull().sum().sum()}")

    st.markdown("**Distribusi User Type**")
    ut_count = df['User_Type'].value_counts().reset_index()
    ut_count.columns = ['User Type', 'Jumlah Transaksi']
    st.dataframe(ut_count, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Total Income vs Expense</div>', unsafe_allow_html=True)
    summary = df.groupby('Type')['Amount'].sum()
    fig_s, ax_s = plt.subplots(figsize=(6, 4))
    bars = ax_s.bar(summary.index, summary.values / 1e6,
                    color=['#E24B4A', '#1D9E75'], edgecolor='white', width=0.5)
    ax_s.set_title('Total Income vs Expense', fontsize=13)
    ax_s.set_xlabel('Type'); ax_s.set_ylabel('Amount (juta Rp)')
    for bar, val in zip(bars, summary.values):
        ax_s.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                  f'Rp {val/1e6:.1f}jt', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_s); plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">[BQ2] Total Pengeluaran per Kategori</div>', unsafe_allow_html=True)
    expense_cat = expense_only.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    fig_cat, ax_cat = plt.subplots(figsize=(12, 5))
    bars = ax_cat.bar(expense_cat.index, expense_cat.values / 1e6,
                      color='#7F77DD', edgecolor='white')
    ax_cat.set_title('Total Pengeluaran per Kategori', fontsize=13)
    ax_cat.set_xlabel('Category'); ax_cat.set_ylabel('Total Amount (juta Rp)')
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, expense_cat.values):
        ax_cat.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val/1e6:.1f}jt', ha='center', fontsize=8)
    plt.tight_layout(); st.pyplot(fig_cat); plt.close()

    st.markdown('<div class="section-title">[BQ2] Tren Total Pengeluaran Bulanan</div>', unsafe_allow_html=True)
    monthly_expense = expense_only.groupby(['Year', 'Month'])['Amount'].sum().reset_index()
    monthly_expense['Period'] = monthly_expense['Year'].astype(str) + '-' + \
                                 monthly_expense['Month'].astype(str).str.zfill(2)
    monthly_expense = monthly_expense.sort_values(['Year', 'Month'])
    fig_tren, ax_tren = plt.subplots(figsize=(12, 5))
    ax_tren.plot(monthly_expense['Period'], monthly_expense['Amount'],
                 marker='o', color='#7F77DD', linewidth=2)
    ax_tren.set_title('Tren Total Pengeluaran Bulanan', fontsize=13)
    ax_tren.set_xlabel('Periode'); ax_tren.set_ylabel('Total Amount (Rp)')
    plt.xticks(rotation=45, ha='right'); ax_tren.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig_tren); plt.close()

    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.markdown('<div class="section-title">[BQ1] Pola Harbolnas 11/11</div>', unsafe_allow_html=True)
        harb_1111 = df[(df['Month'] == 11) & (df['Day'] == 11)]
        if len(harb_1111) > 0:
            h1 = harb_1111['Category'].value_counts().head(8)
            fig_h1, ax_h1 = plt.subplots(figsize=(5, 4))
            ax_h1.bar(h1.index, h1.values, color='#E24B4A', edgecolor='white')
            ax_h1.set_title(f'Harbolnas 11/11 — {len(harb_1111)} transaksi')
            ax_h1.set_ylabel('Jumlah Transaksi')
            plt.xticks(rotation=30, ha='right'); plt.tight_layout()
            st.pyplot(fig_h1); plt.close()
        else:
            st.info("Tidak ada data Harbolnas 11/11.")

    with col_eda2:
        st.markdown('<div class="section-title">[BQ1] Pola Harbolnas 12/12</div>', unsafe_allow_html=True)
        harb_1212 = df[(df['Month'] == 12) & (df['Day'] == 12)]
        if len(harb_1212) > 0:
            h2 = harb_1212['Category'].value_counts().head(8)
            fig_h2, ax_h2 = plt.subplots(figsize=(5, 4))
            ax_h2.bar(h2.index, h2.values, color='#FAC775', edgecolor='white')
            ax_h2.set_title(f'Harbolnas 12/12 — {len(harb_1212)} transaksi')
            ax_h2.set_ylabel('Jumlah Transaksi')
            plt.xticks(rotation=30, ha='right'); plt.tight_layout()
            st.pyplot(fig_h2); plt.close()
        else:
            st.info("Tidak ada data Harbolnas 12/12.")

    st.markdown('<div class="section-title">[BQ2] Pola Pengeluaran Ramadan (Bln 3–4)</div>', unsafe_allow_html=True)
    ramadan_avg = df[df['Month'].isin([3, 4])].groupby('Category')['Amount'].mean().sort_values(ascending=False)
    fig_ram, ax_ram = plt.subplots(figsize=(12, 4))
    ax_ram.bar(ramadan_avg.index, ramadan_avg.values / 1e3, color='#FAC775', edgecolor='white')
    ax_ram.set_title('Rata-rata Pengeluaran per Kategori Saat Ramadan', fontsize=13)
    ax_ram.set_ylabel('Rata-rata Amount (ribu Rp)')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    st.pyplot(fig_ram); plt.close()

    st.markdown('<div class="section-title">[BQ1] User dengan Total Pengeluaran Terbesar</div>', unsafe_allow_html=True)
    boros_user = expense_only.groupby('User_ID')['Amount'].sum().sort_values(ascending=False).head(10)
    fig_boros, ax_boros = plt.subplots(figsize=(10, 4))
    ax_boros.bar(boros_user.index, boros_user.values / 1e6, color='#E24B4A', edgecolor='white')
    ax_boros.set_title('Top 10 User — Total Pengeluaran Terbesar', fontsize=13)
    ax_boros.set_ylabel('Total Amount (juta Rp)')
    for i, (uid, val) in enumerate(boros_user.items()):
        ax_boros.text(i, val/1e6 + 0.2, f'{val/1e6:.1f}jt', ha='center', fontsize=8)
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    st.pyplot(fig_boros); plt.close()

    st.markdown('<div class="section-title">[BQ2] Distribusi Pengeluaran per User Type (Boxplot)</div>', unsafe_allow_html=True)
    ut_order = [u for u in ['hemat', 'normal', 'anak kos', 'hustler', 'boros']
                if u in expense_only['User_Type'].unique()]
    fig_box, ax_box = plt.subplots(figsize=(10, 5))
    bp = ax_box.boxplot(
        [expense_only[expense_only['User_Type'] == u]['Amount'].values for u in ut_order],
        labels=ut_order, patch_artist=True)
    colors_box = ['#1D9E75', '#7F77DD', '#FAC775', '#5DCAA5', '#E24B4A']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax_box.set_title('Distribusi Pengeluaran per User Type', fontsize=13)
    ax_box.set_ylabel('Amount (Rp)'); ax_box.grid(alpha=0.3)
    plt.tight_layout(); st.pyplot(fig_box); plt.close()

    st.markdown("**Rata-rata pengeluaran per User Type:**")
    avg_per_type = expense_only.groupby('User_Type')['Amount'].mean().sort_values(ascending=False)
    avg_df = pd.DataFrame({'User Type': avg_per_type.index,
                           'Rata-rata per Transaksi': avg_per_type.apply(lambda x: f'Rp {x:,.0f}')})
    st.dataframe(avg_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">[BQ2] Heatmap Korelasi Fitur Numerik</div>', unsafe_allow_html=True)
    num_cols = ['Amount', 'amount_log', 'Month', 'Day', 'day_of_week',
                'is_weekday', 'is_ramadan', 'is_harbolnas', 'is_income']
    corr_matrix = df[num_cols].corr().round(3)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, linewidths=0.5, ax=ax_corr)
    ax_corr.set_title('Heatmap Korelasi Fitur Numerik', fontsize=14)
    plt.tight_layout(); st.pyplot(fig_corr); plt.close()

    st.markdown("**Korelasi Amount vs fitur lain:**")
    corr_amount = corr_matrix['Amount'].sort_values(ascending=False).reset_index()
    corr_amount.columns = ['Fitur', 'Korelasi dengan Amount']
    st.dataframe(corr_amount, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">[BQ2] Tren Pengeluaran Bulanan per User Type</div>', unsafe_allow_html=True)
    mt = expense_only.groupby(['Year', 'Month', 'User_Type'])['Amount'].sum().reset_index()
    mt['Period'] = mt['Year'].astype(str) + '-' + mt['Month'].astype(str).str.zfill(2)
    mt = mt.sort_values('Period')
    fig_mt, ax_mt = plt.subplots(figsize=(12, 5))
    palette = {'hemat': '#1D9E75', 'normal': '#7F77DD', 'anak kos': '#FAC775',
               'hustler': '#5DCAA5', 'boros': '#E24B4A'}
    for utype in mt['User_Type'].unique():
        s = mt[mt['User_Type'] == utype]
        ax_mt.plot(s['Period'], s['Amount'] / 1e6, marker='o', label=utype,
                   color=palette.get(utype, '#888'), linewidth=1.5)
    ax_mt.set_title('Rata-rata Total Pengeluaran Bulanan per User Type', fontsize=13)
    ax_mt.set_ylabel('Total Expense (juta Rp)'); ax_mt.legend(); ax_mt.grid(alpha=0.3)
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    st.pyplot(fig_mt); plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 3 — A/B TESTING
# ══════════════════════════════════════════════════════════════
with tab3:
    # ── Test 1: Mann-Whitney boros vs hemat ──────────────────
    st.markdown("#### Test 1 — Mann-Whitney U: Pengeluaran 'boros' vs 'hemat'")
    st.markdown("""
    <div class="info-box">
    <b>H0:</b> Tidak ada perbedaan pengeluaran antara user boros dan hemat<br>
    <b>H1:</b> Ada perbedaan signifikan pengeluaran antara user boros dan hemat
    </div>
    """, unsafe_allow_html=True)

    boros_exp = expense_only[expense_only['User_Type'] == 'boros']['Amount']
    hemat_exp = expense_only[expense_only['User_Type'] == 'hemat']['Amount']
    u1, p1 = mannwhitneyu(boros_exp, hemat_exp, alternative='two-sided')
    sig1 = p1 < 0.05

    ct1a, ct1b = st.columns(2)
    with ct1a:
        st.markdown(f"""<div class="ab-result-box">
        <b>Rata-rata pengeluaran boros :</b> Rp {boros_exp.mean():,.0f}<br>
        <b>Rata-rata pengeluaran hemat :</b> Rp {hemat_exp.mean():,.0f}<br>
        <b>U-Statistic :</b> {u1:,.0f}<br>
        <b>p-value :</b> {p1:.6f}<br><br>
        <b>KESIMPULAN:</b> <span class="{'ab-sig' if sig1 else 'ab-nosig'}">
        {'✅ SIGNIFIKAN (p < 0.05) — tolak H0. Terdapat perbedaan pengeluaran yang signifikan antara user boros dan hemat.'
         if sig1 else '❌ TIDAK SIGNIFIKAN (p ≥ 0.05) — gagal tolak H0'}</span>
        </div>""", unsafe_allow_html=True)
    with ct1b:
        fig_ab1, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].boxplot([boros_exp.values, hemat_exp.values], labels=['boros', 'hemat'],
                        patch_artist=True, boxprops=dict(facecolor='#E24B4A', alpha=0.6))
        axes[0].set_title('Distribusi Pengeluaran:\nBoros vs Hemat')
        axes[0].set_ylabel('Amount (Rp)')
        means = [boros_exp.mean(), hemat_exp.mean()]
        axes[1].bar(['boros', 'hemat'], means, color=['#E24B4A', '#1D9E75'], alpha=0.8)
        axes[1].set_title('Rata-rata Pengeluaran\nper Transaksi')
        axes[1].set_ylabel('Rata-rata Amount (Rp)')
        for i, v in enumerate(means):
            axes[1].text(i, v + 500, f'Rp {v:,.0f}', ha='center', fontsize=8)
        plt.tight_layout(); st.pyplot(fig_ab1); plt.close()

    st.markdown("---")

    # ── Test 2: Chi-Square ───────────────────────────────────
    st.markdown("#### Test 2 — Chi-Square: Distribusi Kategori vs User Type")
    st.markdown("""
    <div class="info-box">
    <b>H0:</b> Distribusi kategori transaksi tidak dipengaruhi oleh User_Type<br>
    <b>H1:</b> Ada hubungan signifikan antara kategori dan User_Type
    </div>
    """, unsafe_allow_html=True)

    contingency = pd.crosstab(df['User_Type'], df['Category'])
    chi2_val, p_chi, dof, _ = chi2_contingency(contingency)
    sig2 = p_chi < 0.05

    ct2a, ct2b = st.columns(2)
    with ct2a:
        st.markdown(f"""<div class="ab-result-box">
        <b>Chi2-Statistic :</b> {chi2_val:,.4f}<br>
        <b>p-value :</b> {p_chi:.6f}<br>
        <b>Degrees of Freedom :</b> {dof}<br><br>
        <b>KESIMPULAN:</b> <span class="{'ab-sig' if sig2 else 'ab-nosig'}">
        {'✅ SIGNIFIKAN (p < 0.05) — tolak H0. Distribusi kategori transaksi berbeda secara signifikan antar User_Type.'
         if sig2 else '❌ TIDAK SIGNIFIKAN (p ≥ 0.05) — gagal tolak H0'}</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("**Contingency Table (sample):**")
        st.dataframe(contingency, use_container_width=True)
    with ct2b:
        fig_ab2, ax_ab2 = plt.subplots(figsize=(8, 4))
        sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd',
                    linewidths=0.5, ax=ax_ab2)
        ax_ab2.set_title('Frekuensi Transaksi per Kategori & User Type', fontsize=11)
        plt.tight_layout(); st.pyplot(fig_ab2); plt.close()

    st.markdown("---")

    # ── Test 3: Ramadan vs Normal ────────────────────────────
    st.markdown("#### Test 3 — Mann-Whitney U: Efek Ramadan pada Pengeluaran 'makan'")
    st.markdown("""
    <div class="info-box">
    <b>H0:</b> Tidak ada perbedaan pengeluaran makan saat Ramadan vs Normal<br>
    <b>H1:</b> Ada perbedaan signifikan pengeluaran makan saat Ramadan
    </div>
    """, unsafe_allow_html=True)

    makan_df      = expense_only[expense_only['Category'] == 'makan'].copy()
    makan_ramadan = makan_df[makan_df['Month'].isin([3, 4])]['Amount']
    makan_normal  = makan_df[~makan_df['Month'].isin([3, 4])]['Amount']
    u2, p2 = mannwhitneyu(makan_ramadan, makan_normal, alternative='two-sided')
    kenaikan_pct = (makan_ramadan.mean() - makan_normal.mean()) / makan_normal.mean() * 100
    sig3 = p2 < 0.05

    ct3a, ct3b = st.columns(2)
    with ct3a:
        st.markdown(f"""<div class="ab-result-box">
        <b>Rata-rata makan Ramadan :</b> Rp {makan_ramadan.mean():,.0f}<br>
        <b>Rata-rata makan Normal  :</b> Rp {makan_normal.mean():,.0f}<br>
        <b>Kenaikan                :</b> {kenaikan_pct:.1f}%<br>
        <b>U-Statistic             :</b> {u2:,.0f}<br>
        <b>p-value                 :</b> {p2:.6f}<br><br>
        <b>KESIMPULAN:</b> <span class="{'ab-sig' if sig3 else 'ab-nosig'}">
        {'✅ SIGNIFIKAN — efek Ramadan terbukti secara statistik.'
         if sig3 else '❌ TIDAK SIGNIFIKAN'}</span>
        </div>""", unsafe_allow_html=True)
    with ct3b:
        fig_ab3, ax_ab3 = plt.subplots(figsize=(5, 4))
        ax_ab3.bar(['Ramadan', 'Normal'], [makan_ramadan.mean(), makan_normal.mean()],
                   color=['#FAC775', '#7F77DD'], alpha=0.85, edgecolor='white')
        ax_ab3.set_title(f'Efek Ramadan pada Pengeluaran Makan\n(+{kenaikan_pct:.1f}%)', fontsize=11)
        ax_ab3.set_ylabel('Rata-rata Amount (Rp)')
        for i, v in enumerate([makan_ramadan.mean(), makan_normal.mean()]):
            ax_ab3.text(i, v + 500, f'Rp {v:,.0f}', ha='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig_ab3); plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 4 — MODEL RF
# ══════════════════════════════════════════════════════════════
with tab4:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    st.markdown('<div class="section-title">Model Random Forest — Auto-Kategorisasi Transaksi</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Tujuan:</b> Auto-kategorisasi transaksi (target: Category) menggunakan 10 fitur.<br>
    <b>Fitur:</b> Amount, amount_log, is_income, Month, Day, day_of_week, is_weekday,
    is_ramadan, is_harbolnas, User_Type_enc<br>
    <b>Split:</b> Train 80% / Test 20% (stratified)
    </div>
    """, unsafe_allow_html=True)

    model_loaded = False
    if os.path.exists(os.path.join(BASE_DIR, 'rf_model.pkl')) and os.path.exists(os.path.join(BASE_DIR, 'le_user_type.pkl')):
        try:
            rf_model = joblib.load(os.path.join(BASE_DIR, 'rf_model.pkl'))
            le_user  = joblib.load(os.path.join(BASE_DIR, 'le_user_type.pkl'))
            model_loaded = True
            st.success("✅ Model RF berhasil dimuat dari rf_model.pkl")
        except Exception as e:
            st.warning(f"⚠️ Gagal load model: {e}")
    else:
        st.warning("⚠️ File `rf_model.pkl` tidak ditemukan. Klik tombol di bawah untuk melatih.")

    if not model_loaded:
        if st.button("🚀 Latih Model RF Sekarang"):
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            df_model = df.copy()
            le_ut = LabelEncoder()
            df_model['User_Type_enc'] = le_ut.fit_transform(df_model['User_Type'])
            FEATURES = ['Amount', 'amount_log', 'is_income', 'Month', 'Day',
                        'day_of_week', 'is_weekday', 'is_ramadan', 'is_harbolnas', 'User_Type_enc']
            X = df_model[FEATURES]; y = df_model['Category']
            data_m = pd.concat([X, y], axis=1).dropna()
            X = data_m[FEATURES]; y = data_m['Category']
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            with st.spinner("Melatih RandomForest (20 trees)..."):
                rf = RandomForestClassifier(n_estimators=20, max_depth=15, random_state=42, n_jobs=-1)
                rf.fit(X_tr, y_tr)
                y_pred = rf.predict(X_te)
                acc = accuracy_score(y_te, y_pred)
            joblib.dump(rf, os.path.join(BASE_DIR, 'rf_model.pkl')); joblib.dump(le_ut, os.path.join(BASE_DIR, 'le_user_type.pkl'))
            st.success(f"✅ Model dilatih! Accuracy: {acc*100:.2f}%")
            st.text(classification_report(y_te, y_pred))
            st.rerun()
    else:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        df_model = df.copy()
        le_ut2 = LabelEncoder()
        df_model['User_Type_enc'] = le_ut2.fit_transform(df_model['User_Type'])
        FEATURES = ['Amount', 'amount_log', 'is_income', 'Month', 'Day',
                    'day_of_week', 'is_weekday', 'is_ramadan', 'is_harbolnas', 'User_Type_enc']
        X = df_model[FEATURES]; y = df_model['Category']
        data_m = pd.concat([X, y], axis=1).dropna()
        X = data_m[FEATURES]; y = data_m['Category']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred = rf_model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)

        st.markdown(f"**Accuracy: {acc*100:.2f}%** | Train: {len(X_tr):,} | Test: {len(X_te):,}")

        col_rf1, col_rf2 = st.columns(2)
        with col_rf1:
            st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y_te, y_pred, labels=rf_model.classes_)
            cm_df = pd.DataFrame(cm, index=rf_model.classes_, columns=rf_model.classes_)
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5, ax=ax_cm)
            ax_cm.set_title(f'Confusion Matrix — Accuracy: {acc*100:.2f}%', fontsize=12)
            ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
            plt.tight_layout(); st.pyplot(fig_cm); plt.close()

        with col_rf2:
            st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
            feat_imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
            fig_fi, ax_fi = plt.subplots(figsize=(6, 5))
            feat_imp.plot(kind='bar', color='#7F77DD', edgecolor='white', ax=ax_fi)
            ax_fi.set_title('Feature Importance — Random Forest', fontsize=12)
            ax_fi.set_ylabel('Importance Score')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout(); st.pyplot(fig_fi); plt.close()

            st.markdown("**Feature Importance (detail):**")
            fi_df = pd.DataFrame({'Fitur': feat_imp.index, 'Importance': feat_imp.values.round(4)})
            st.dataframe(fi_df, use_container_width=True, hide_index=True)
        
# ══════════════════════════════════════════════════════════════
# TAB 5 — LSTM
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">LSTM — Prediksi Total Pengeluaran Bulanan per User</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Tujuan:</b> Prediksi total pengeluaran bulanan per user berbasis time-series LSTM.<br>
    <b>Pipeline:</b> Agregasi bulanan → MinMaxScaler → Sliding window sequence → LSTM → Evaluasi MAE/RMSE/sMAPE<br>
    <b>Target MAE:</b> ≤ Rp 300.000 | <b>Target RMSE:</b> ≤ Rp 450.000 | <b>Target sMAPE:</b> ≤ 25%<br>
    <b>Split:</b> Train 70% / Val 15% / Test 15% (per user, time-based)
    </div>
    """, unsafe_allow_html=True)

    # Agregasi bulanan
    exp_df_lstm = df[df['Type'] == 'Expense'].copy()
    inc_df_lstm = df[df['Type'] == 'Income'].copy()

    monthly_exp = exp_df_lstm.groupby(['User_ID', 'User_Type', 'Year', 'Month']).agg(
        total_expense=('Amount', 'sum'),
        frekuensi_exp=('Amount', 'count'),
        avg_expense  =('Amount', 'mean'),
        max_expense  =('Amount', 'max'),
    ).reset_index()

    monthly_inc = inc_df_lstm.groupby(['User_ID', 'Year', 'Month']).agg(
        total_income =('Amount', 'sum'),
        frekuensi_inc=('Amount', 'count'),
    ).reset_index()

    monthly = pd.merge(monthly_exp, monthly_inc, on=['User_ID', 'Year', 'Month'], how='left').fillna(0)
    monthly['net']                = monthly['total_income'] - monthly['total_expense']
    monthly['is_ramadan']         = monthly['Month'].isin([3, 4]).astype(int)
    monthly['is_harbolnas_month'] = monthly['Month'].isin([11, 12]).astype(int)
    monthly['Period']             = monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2)
    monthly = monthly.sort_values(['User_ID', 'Year', 'Month']).reset_index(drop=True)

    st.markdown(f"**Dataset bulanan:** {monthly.shape[0]:,} baris | {monthly['User_ID'].nunique()} user | "
                f"Periode: {monthly['Period'].min()} s/d {monthly['Period'].max()}")

    st.markdown('<div class="section-title">Visualisasi Tren Bulanan per User Type</div>', unsafe_allow_html=True)
    fig_lstm, axes_lstm = plt.subplots(2, 1, figsize=(14, 8))

    avg_monthly = monthly.groupby('Period')['total_expense'].mean().reset_index().sort_values('Period')
    axes_lstm[0].plot(avg_monthly['Period'], avg_monthly['total_expense'],
                      marker='o', color='#E24B4A', linewidth=2)
    axes_lstm[0].set_title('Rata-rata Total Pengeluaran Bulanan (Semua User)', fontsize=13)
    axes_lstm[0].set_ylabel('Total Expense (Rp)'); axes_lstm[0].grid(alpha=0.3)
    axes_lstm[0].tick_params(axis='x', rotation=45)

    palette_lstm = {'hemat': '#1D9E75', 'normal': '#7F77DD', 'anak kos': '#FAC775',
                    'hustler': '#5DCAA5', 'boros': '#E24B4A'}
    for utype in monthly['User_Type'].unique():
        sub = monthly[monthly['User_Type'] == utype].groupby('Period')['total_expense'].mean()
        axes_lstm[1].plot(sub.index, sub.values, marker='o', label=utype,
                          color=palette_lstm.get(utype, '#888'), linewidth=1.5)
    axes_lstm[1].set_title('Rata-rata Total Pengeluaran Bulanan per User Type', fontsize=13)
    axes_lstm[1].set_ylabel('Total Expense (Rp)'); axes_lstm[1].legend(); axes_lstm[1].grid(alpha=0.3)
    axes_lstm[1].tick_params(axis='x', rotation=45)
    plt.tight_layout(); st.pyplot(fig_lstm); plt.close()

    st.markdown('<div class="section-title">Statistik Dataset Bulanan</div>', unsafe_allow_html=True)
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("**Statistik total_expense:**")
        stat_df = monthly['total_expense'].describe().apply(lambda x: f'Rp {x:,.0f}').reset_index()
        stat_df.columns = ['Statistik', 'Nilai']
        st.dataframe(stat_df, use_container_width=True, hide_index=True)
    with col_l2:
        st.markdown("**Jumlah bulan aktif per user:**")
        bulan_per_user = monthly.groupby('User_ID').size().describe().apply(lambda x: f'{x:.1f}').reset_index()
        bulan_per_user.columns = ['Statistik', 'Nilai']
        st.dataframe(bulan_per_user, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Konfigurasi LSTM</div>', unsafe_allow_html=True)
    lstm_config = pd.DataFrame({
        'Parameter': ['LOOKBACK', 'LSTM Units (L1)', 'LSTM Units (L2)', 'Dropout',
                      'Batch Size', 'Max Epochs', 'Early Stopping Patience',
                      'Target MAE', 'Target RMSE', 'Target sMAPE'],
        'Nilai': ['2 bulan', '64', '32', '0.2', '16', '100', '15 epoch',
                  '≤ Rp 300.000', '≤ Rp 450.000', '≤ 25%']
    })
    st.dataframe(lstm_config, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-box">
    ⚠️ Training LSTM membutuhkan GPU/CPU yang cukup dan tidak dijalankan langsung di dashboard ini.
    Model LSTM dilatih di Google Colab dan hasilnya disimpan ke file <code>lstm_model_final.keras</code>,
    <code>scaler.pkl</code>, dan <code>scaler_target.pkl</code>.
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists('lstm_model_final.keras') or os.path.exists('lstm_model_final.h5'):
        st.success("✅ File model LSTM ditemukan di direktori ini.")
    else:
        st.warning("⚠️ File model LSTM tidak ditemukan. Upload lstm_model_final.keras ke folder dashboard.")


# ══════════════════════════════════════════════════════════════
# TAB 6 — DATA DICTIONARY
# ══════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">Data Dictionary — Dataset Asli</div>', unsafe_allow_html=True)
    dd_orig = pd.DataFrame({
        'Kolom'      : ['User_ID', 'User_Type', 'Date', 'Category', 'Amount', 'Type', 'Year', 'Month', 'Day'],
        'Tipe Data'  : ['object', 'object', 'datetime64', 'object', 'int64', 'object', 'int64', 'int64', 'int64'],
        'Deskripsi'  : [
            'ID unik setiap pengguna (U001–U050)',
            'Kepribadian finansial: hemat, normal, boros, hustler, anak kos',
            'Tanggal transaksi (format YYYY-MM-DD)',
            'Kategori transaksi: makan, transport, belanja online, dst.',
            'Nominal transaksi dalam Rupiah (positif)',
            'Jenis transaksi: Income (pemasukan) atau Expense (pengeluaran)',
            'Tahun transaksi, diturunkan dari kolom Date',
            'Bulan transaksi (1–12), diturunkan dari Date',
            'Tanggal dalam bulan (1–31), diturunkan dari Date'
        ],
        'Contoh Nilai': ['U001', 'boros', '2025-03-14', 'makan', '35000', 'Expense', '2025', '3', '14']
    })
    st.dataframe(dd_orig, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Data Dictionary — Fitur Engineering</div>', unsafe_allow_html=True)
    dd_feat = pd.DataFrame({
        'Kolom'      : ['day_of_week', 'is_weekday', 'is_ramadan', 'is_harbolnas', 'amount_log', 'is_income'],
        'Tipe Data'  : ['int64', 'int64', 'int64', 'int64', 'float64', 'int64'],
        'Deskripsi'  : [
            'Hari dalam minggu (0=Senin, 6=Minggu), diturunkan dari Date',
            'Binary: 1 jika hari kerja (Senin–Jumat)',
            'Binary: 1 jika bulan Ramadan (Maret atau April)',
            'Binary: 1 jika tanggal Harbolnas (11/11 atau 12/12)',
            'Transformasi log natural dari Amount: log1p(Amount)',
            'Binary: 1 jika Income, 0 jika Expense'
        ],
        'Contoh Nilai': ['1', '1', '1', '0', '10.46', '0']
    })
    st.dataframe(dd_feat, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Preview Data (500 baris pertama)</div>', unsafe_allow_html=True)
    st.dataframe(df_f.head(500), use_container_width=True)

    st.markdown('<div class="section-title">Statistik Deskriptif</div>', unsafe_allow_html=True)
    st.dataframe(df_f[['Amount', 'amount_log', 'Month', 'Day']].describe().round(2),
                 use_container_width=True)