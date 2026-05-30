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

st.set_page_config(page_title="GenZCash Dashboard", page_icon="💸", layout="wide")

st.markdown("""
<style>
.section-title{font-size:18px;font-weight:600;color:#1a1a1a;margin:24px 0 12px;
  padding-bottom:6px;border-bottom:2px solid #7F77DD}
.ab-result-box{background:#f0f4ff;border-radius:10px;padding:14px 18px;
  margin-bottom:10px;border:1px solid #d0d8ff}
.ab-sig{color:#0F6E56;font-weight:600}
.ab-nosig{color:#A32D2D;font-weight:600}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('genz_financial_dataset_v3.csv')
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

# ── SIDEBAR ──────────────────────────────────────────────────
st.sidebar.markdown("## 💸 GenZCash")
st.sidebar.markdown("---")
st.sidebar.markdown("### Filter Data")
user_types = st.sidebar.multiselect(
    "User Type", sorted(df['User_Type'].unique()), default=sorted(df['User_Type'].unique()))
years = st.sidebar.multiselect(
    "Tahun", sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
tipe_tx = st.sidebar.radio("Tipe Transaksi", ["Semua", "Income", "Expense"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Coding Camp 2026 — Data Science Project")

df_f = df[df['User_Type'].isin(user_types) & df['Year'].isin(years)]
if tipe_tx != "Semua":
    df_f = df_f[df_f['Type'] == tipe_tx]
df_exp = df_f[df_f['Type'] == 'Expense'].copy()
df_inc = df_f[df_f['Type'] == 'Income'].copy()

# ── HEADER ───────────────────────────────────────────────────
st.title("💸SPENDIGO")
st.caption(
    f"Dataset: {len(df_f):,} transaksi | {df_f['User_ID'].nunique()} user | "
    f"{df_f['Date'].min().date()} s/d {df_f['Date'].max().date()}"
)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🔍 EDA", "🧪 A/B Testing", "🤖 Model RF", "📋 Data"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">KPI Utama</div>', unsafe_allow_html=True)
    total_inc = df_inc['Amount'].sum()
    total_exp = df_exp['Amount'].sum()
    net_val   = total_inc - total_exp
    avg_tx    = df_exp['Amount'].mean() if len(df_exp) > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Income",    f"Rp {total_inc/1e6:.1f}jt")
    c2.metric("💸 Total Expense",   f"Rp {total_exp/1e6:.1f}jt")
    c3.metric("📈 Net Balance",     f"Rp {net_val/1e6:.1f}jt",
              delta="surplus" if net_val > 0 else "defisit")
    c4.metric("🔢 Total Transaksi", f"{len(df_f):,}")
    c5.metric("📊 Avg Expense/Tx",  f"Rp {avg_tx:,.0f}")

    st.markdown('<div class="section-title">Income vs Expense per Bulan</div>', unsafe_allow_html=True)
    mp = df_f.groupby(['Year', 'Month', 'Type'])['Amount'].sum().reset_index()
    mp['Period'] = mp['Year'].astype(str) + '-' + mp['Month'].astype(str).str.zfill(2)
    mp = mp.sort_values('Period')
    fig, ax = plt.subplots(figsize=(12, 4))
    for t, c in [('Income', '#1D9E75'), ('Expense', '#E24B4A')]:
        s = mp[mp['Type'] == t]
        ax.plot(s['Period'], s['Amount'] / 1e6, marker='o', label=t, color=c, linewidth=2)
    ax.set_ylabel('Jumlah (juta Rp)'); ax.legend(); ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(); st.pyplot(fig); plt.close()

    cl, cr = st.columns(2)
    with cl:
        st.markdown('<div class="section-title">Distribusi User Type</div>', unsafe_allow_html=True)
        uc = df[['User_ID', 'User_Type']].drop_duplicates()['User_Type'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie(uc, labels=uc.index, autopct='%1.0f%%',
                colors=['#7F77DD', '#1D9E75', '#E24B4A', '#FAC775', '#5DCAA5'], startangle=90)
        ax2.set_title('Komposisi User Type')
        plt.tight_layout(); st.pyplot(fig2); plt.close()
    with cr:
        st.markdown('<div class="section-title">Top 5 Kategori Pengeluaran</div>', unsafe_allow_html=True)
        tc = df_exp.groupby('Category')['Amount'].sum().sort_values(ascending=True).tail(5)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        bars = ax3.barh(tc.index, tc.values / 1e6, color='#7F77DD', edgecolor='white')
        ax3.set_xlabel('Total (juta Rp)'); ax3.set_title('Top 5 Kategori Pengeluaran')
        for bar, val in zip(bars, tc.values):
            ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f'Rp {val/1e6:.1f}jt', va='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig3); plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Distribusi Pengeluaran per User Type</div>', unsafe_allow_html=True)
    ut_order = [u for u in ['hemat', 'normal', 'anak kos', 'hustler', 'boros']
                if u in df_exp['User_Type'].unique()]
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    bp = ax4.boxplot([df_exp[df_exp['User_Type'] == u]['Amount'].values for u in ut_order],
                     labels=ut_order, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#1D9E75', '#7F77DD', '#FAC775', '#E24B4A', '#5DCAA5']):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax4.set_ylabel('Amount (Rp)'); ax4.set_title('Sebaran Pengeluaran per User Type')
    ax4.grid(alpha=0.3); plt.tight_layout(); st.pyplot(fig4); plt.close()

    ce1, ce2 = st.columns(2)
    with ce1:
        st.markdown('<div class="section-title">Pola Pengeluaran Ramadan</div>', unsafe_allow_html=True)
        ram_avg = (df_exp[df_exp['is_ramadan'] == 1]
                   .groupby('Category')['Amount'].mean()
                   .sort_values(ascending=False).head(6))
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        ax5.bar(ram_avg.index, ram_avg.values / 1e3, color='#FAC775', edgecolor='white')
        ax5.set_ylabel('Rata-rata (ribu Rp)'); ax5.set_title('Avg Pengeluaran Saat Ramadan')
        plt.xticks(rotation=30, ha='right'); plt.tight_layout(); st.pyplot(fig5); plt.close()
    with ce2:
        st.markdown('<div class="section-title">Pola Harbolnas</div>', unsafe_allow_html=True)
        harb = df_exp[df_exp['is_harbolnas'] == 1]
        if len(harb) > 0:
            hc = harb['Category'].value_counts().head(6)
            fig6, ax6 = plt.subplots(figsize=(5, 4))
            ax6.bar(hc.index, hc.values, color='#E24B4A', edgecolor='white')
            ax6.set_ylabel('Jumlah Transaksi'); ax6.set_title('Frekuensi Transaksi Harbolnas')
            plt.xticks(rotation=30, ha='right'); plt.tight_layout(); st.pyplot(fig6); plt.close()
        else:
            st.info("Tidak ada data Harbolnas pada filter ini.")

    st.markdown('<div class="section-title">Heatmap Korelasi</div>', unsafe_allow_html=True)
    num_cols = ['Amount', 'amount_log', 'Month', 'Day', 'day_of_week',
                'is_weekday', 'is_ramadan', 'is_harbolnas', 'is_income']
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    sns.heatmap(df_f[num_cols].corr().round(2), annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, linewidths=0.5, ax=ax7)
    ax7.set_title('Heatmap Korelasi Fitur Numerik')
    plt.tight_layout(); st.pyplot(fig7); plt.close()

    st.markdown('<div class="section-title">Tren Bulanan per User Type</div>', unsafe_allow_html=True)
    mt = df_exp.groupby(['Year', 'Month', 'User_Type'])['Amount'].sum().reset_index()
    mt['Period'] = mt['Year'].astype(str) + '-' + mt['Month'].astype(str).str.zfill(2)
    mt = mt.sort_values('Period')
    fig8, ax8 = plt.subplots(figsize=(12, 4))
    palette = {'hemat': '#1D9E75', 'normal': '#7F77DD', 'anak kos': '#FAC775',
               'hustler': '#5DCAA5', 'boros': '#E24B4A'}
    for utype in mt['User_Type'].unique():
        s = mt[mt['User_Type'] == utype]
        ax8.plot(s['Period'], s['Amount'] / 1e6, marker='o', label=utype,
                 color=palette.get(utype, '#888'), linewidth=1.5)
    ax8.set_ylabel('Total Expense (juta Rp)'); ax8.legend(); ax8.grid(alpha=0.3)
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(); st.pyplot(fig8); plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 3 — A/B TESTING
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">A/B Testing — Uji Statistik</div>', unsafe_allow_html=True)
    exp_all = df[df['Type'] == 'Expense'].copy()

    # Test 1 — Mann-Whitney boros vs hemat
    st.markdown("#### Test 1 — Mann-Whitney U: Boros vs Hemat")
    boros_exp = exp_all[exp_all['User_Type'] == 'boros']['Amount']
    hemat_exp = exp_all[exp_all['User_Type'] == 'hemat']['Amount']
    u1, p1 = mannwhitneyu(boros_exp, hemat_exp, alternative='two-sided')
    sig1 = p1 < 0.05
    ct1a, ct1b = st.columns(2)
    with ct1a:
        st.markdown(f"""<div class="ab-result-box">
        <b>H0:</b> Tidak ada perbedaan pengeluaran boros vs hemat<br>
        <b>U-statistic:</b> {u1:,.0f} &nbsp;|&nbsp; <b>p-value:</b> {p1:.6f}<br>
        <b>Kesimpulan:</b> <span class="{'ab-sig' if sig1 else 'ab-nosig'}">
        {'✅ SIGNIFIKAN — tolak H0' if sig1 else '❌ TIDAK SIGNIFIKAN'}</span>
        </div>""", unsafe_allow_html=True)
        st.metric("Rata-rata Boros", f"Rp {boros_exp.mean():,.0f}")
        st.metric("Rata-rata Hemat", f"Rp {hemat_exp.mean():,.0f}")
    with ct1b:
        fig_ab1, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].boxplot([boros_exp.values, hemat_exp.values], labels=['boros', 'hemat'],
                        patch_artist=True, boxprops=dict(facecolor='#E24B4A', alpha=0.6))
        axes[0].set_title('Distribusi'); axes[0].set_ylabel('Amount (Rp)')
        means = [boros_exp.mean(), hemat_exp.mean()]
        axes[1].bar(['boros', 'hemat'], means, color=['#E24B4A', '#1D9E75'], alpha=0.8)
        axes[1].set_title('Rata-rata')
        for i, v in enumerate(means):
            axes[1].text(i, v + 500, f'Rp {v:,.0f}', ha='center', fontsize=8)
        plt.tight_layout(); st.pyplot(fig_ab1); plt.close()

    st.markdown("---")

    # Test 2 — Chi-Square
    st.markdown("#### Test 2 — Chi-Square: Kategori vs User Type")
    ctab = pd.crosstab(exp_all['User_Type'], exp_all['Category'])
    chi2, p2, dof, _ = chi2_contingency(ctab)
    sig2 = p2 < 0.05
    ct2a, ct2b = st.columns(2)
    with ct2a:
        st.markdown(f"""<div class="ab-result-box">
        <b>H0:</b> Distribusi kategori tidak dipengaruhi User_Type<br>
        <b>Chi2:</b> {chi2:,.4f} &nbsp;|&nbsp; <b>p-value:</b> {p2:.6f} &nbsp;|&nbsp; <b>dof:</b> {dof}<br>
        <b>Kesimpulan:</b> <span class="{'ab-sig' if sig2 else 'ab-nosig'}">
        {'✅ SIGNIFIKAN — tolak H0' if sig2 else '❌ TIDAK SIGNIFIKAN'}</span>
        </div>""", unsafe_allow_html=True)
    with ct2b:
        fig_ab2, ax_ab2 = plt.subplots(figsize=(7, 3))
        sns.heatmap(ctab, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5, ax=ax_ab2)
        ax_ab2.set_title('Frekuensi Transaksi per Kategori & User Type')
        plt.tight_layout(); st.pyplot(fig_ab2); plt.close()

    st.markdown("---")

    # Test 3 — Ramadan
    st.markdown("#### Test 3 — Mann-Whitney U: Efek Ramadan pada Pengeluaran Makan")
    makan_df  = exp_all[exp_all['Category'] == 'makan'].copy()
    makan_ram = makan_df[makan_df['Month'].isin([3, 4])]['Amount']
    makan_nor = makan_df[~makan_df['Month'].isin([3, 4])]['Amount']
    u3, p3 = mannwhitneyu(makan_ram, makan_nor, alternative='two-sided')
    kenaikan = (makan_ram.mean() - makan_nor.mean()) / makan_nor.mean() * 100
    sig3 = p3 < 0.05
    ct3a, ct3b = st.columns(2)
    with ct3a:
        st.markdown(f"""<div class="ab-result-box">
        <b>H0:</b> Tidak ada perbedaan makan Ramadan vs Normal<br>
        <b>Kenaikan:</b> {kenaikan:.1f}% &nbsp;|&nbsp; <b>p-value:</b> {p3:.6f}<br>
        <b>Kesimpulan:</b> <span class="{'ab-sig' if sig3 else 'ab-nosig'}">
        {'✅ SIGNIFIKAN — efek Ramadan terbukti' if sig3 else '❌ TIDAK SIGNIFIKAN'}</span>
        </div>""", unsafe_allow_html=True)
        st.metric("Avg Makan Ramadan", f"Rp {makan_ram.mean():,.0f}")
        st.metric("Avg Makan Normal",  f"Rp {makan_nor.mean():,.0f}")
    with ct3b:
        fig_ab3, ax_ab3 = plt.subplots(figsize=(5, 3))
        ax_ab3.bar(['Ramadan', 'Normal'], [makan_ram.mean(), makan_nor.mean()],
                   color=['#FAC775', '#7F77DD'], alpha=0.85)
        ax_ab3.set_ylabel('Rata-rata Amount (Rp)')
        ax_ab3.set_title(f'Efek Ramadan pada Makan (+{kenaikan:.1f}%)')
        plt.tight_layout(); st.pyplot(fig_ab3); plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 4 — MODEL RF
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Model Random Forest — Auto-Kategorisasi</div>',
                unsafe_allow_html=True)

    model_loaded = False
    if os.path.exists('rf_model.pkl') and os.path.exists('le_user_type.pkl'):
        try:
            rf_model = joblib.load('rf_model.pkl')
            le_user  = joblib.load('le_user_type.pkl')
            model_loaded = True
            st.success("✅ Model RF berhasil dimuat")
        except Exception as e:
            st.warning(f"⚠️ Gagal load model: {e}")
    else:
        st.warning("⚠️ File `rf_model.pkl` tidak ditemukan. Klik tombol di bawah untuk melatih.")

    if not model_loaded:
        if st.button("🚀 Latih Model RF Sekarang"):
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report
            df_model = df.copy()
            le_ut = LabelEncoder()
            df_model['User_Type_enc'] = le_ut.fit_transform(df_model['User_Type'])
            FEATURES = ['Amount', 'amount_log', 'is_income', 'Month', 'Day',
                        'day_of_week', 'is_weekday', 'is_ramadan', 'is_harbolnas', 'User_Type_enc']
            X = df_model[FEATURES]; y = df_model['Category']
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            with st.spinner("Melatih RandomForest (200 trees)..."):
                rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                rf.fit(X_tr, y_tr)
                y_pred = rf.predict(X_te)
                acc = accuracy_score(y_te, y_pred)
            joblib.dump(rf, 'rf_model.pkl'); joblib.dump(le_ut, 'le_user_type.pkl')
            st.success(f"✅ Model dilatih! Accuracy: {acc*100:.2f}%")
            st.text(classification_report(y_te, y_pred))
            st.rerun()
    else:
        st.markdown("#### 🔮 Prediksi Kategori Transaksi Baru")
        cp1, cp2 = st.columns(2)
        with cp1:
            p_amount   = st.number_input("Amount (Rp)", min_value=1000, max_value=10000000,
                                          value=50000, step=5000)
            p_type     = st.selectbox("Tipe Transaksi", ["Expense", "Income"])
            p_usertype = st.selectbox("User Type", sorted(df['User_Type'].unique()))
        with cp2:
            p_month   = st.slider("Bulan", 1, 12, 3)
            p_day     = st.slider("Tanggal", 1, 31, 15)
            p_weekday = st.slider("Hari (0=Senin, 6=Minggu)", 0, 6, 1)

        if st.button("🔍 Prediksi Kategori"):
            try:
                ut_enc  = le_user.transform([p_usertype])[0]
                is_inc  = 1 if p_type == 'Income' else 0
                is_wkd  = 1 if p_weekday < 5 else 0
                is_ram  = 1 if p_month in [3, 4] else 0
                is_harb = 1 if (p_month == 11 and p_day == 11) or \
                               (p_month == 12 and p_day == 12) else 0
                X_new = pd.DataFrame([{
                    'Amount': p_amount, 'amount_log': np.log1p(p_amount),
                    'is_income': is_inc, 'Month': p_month, 'Day': p_day,
                    'day_of_week': p_weekday, 'is_weekday': is_wkd,
                    'is_ramadan': is_ram, 'is_harbolnas': is_harb,
                    'User_Type_enc': ut_enc
                }])
                pred_cat  = rf_model.predict(X_new)[0]
                pred_prob = rf_model.predict_proba(X_new)[0]
                st.success(f"✅ Prediksi: **{pred_cat}** (confidence: {pred_prob.max()*100:.1f}%)")
                prob_df = pd.DataFrame({
                    'Kategori': rf_model.classes_,
                    'Probabilitas': pred_prob * 100
                }).sort_values('Probabilitas', ascending=True).tail(6)
                fig_pr, ax_pr = plt.subplots(figsize=(7, 3))
                bars = ax_pr.barh(prob_df['Kategori'], prob_df['Probabilitas'],
                                  color='#7F77DD', edgecolor='white')
                ax_pr.set_xlabel('Probabilitas (%)'); ax_pr.set_title('Distribusi Probabilitas Prediksi')
                for bar, val in zip(bars, prob_df['Probabilitas']):
                    ax_pr.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                               f'{val:.1f}%', va='center', fontsize=9)
                plt.tight_layout(); st.pyplot(fig_pr); plt.close()
            except Exception as e:
                st.error(f"Error prediksi: {e}")

        st.markdown("---")
        st.markdown("#### Feature Importance")
        feat_names = ['Amount', 'amount_log', 'is_income', 'Month', 'Day',
                      'day_of_week', 'is_weekday', 'is_ramadan', 'is_harbolnas', 'User_Type_enc']
        fi = pd.Series(rf_model.feature_importances_, index=feat_names).sort_values(ascending=True)
        fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
        ax_fi.barh(fi.index, fi.values, color='#7F77DD', edgecolor='white')
        ax_fi.set_xlabel('Importance Score'); ax_fi.set_title('Feature Importance — Random Forest')
        plt.tight_layout(); st.pyplot(fig_fi); plt.close()


# ══════════════════════════════════════════════════════════════
# TAB 5 — DATA
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Data Dictionary</div>', unsafe_allow_html=True)
    dd = pd.DataFrame({
        'Kolom'     : ['User_ID', 'User_Type', 'Date', 'Category', 'Amount', 'Type',
                       'Year', 'Month', 'Day', 'is_ramadan', 'is_harbolnas', 'is_weekday', 'amount_log'],
        'Tipe'      : ['object', 'object', 'datetime64', 'object', 'int64', 'object',
                       'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'float64'],
        'Deskripsi' : [
            'ID unik user (U001–U050)',
            'Kepribadian finansial: hemat, normal, boros, hustler, anak kos',
            'Tanggal transaksi (YYYY-MM-DD)',
            'Kategori transaksi: makan, transport, belanja online, dst.',
            'Nominal transaksi dalam Rupiah (positif)',
            'Jenis transaksi: Income atau Expense',
            'Tahun transaksi', 'Bulan (1–12)', 'Tanggal dalam bulan (1–31)',
            '1 jika bulan Ramadan (Maret–April)',
            '1 jika tanggal Harbolnas (11/11 atau 12/12)',
            '1 jika hari kerja (Senin–Jumat)',
            'Transformasi log natural dari Amount'
        ],
        'Contoh'    : ['U001', 'boros', '2025-03-14', 'makan', '35000', 'Expense',
                       '2025', '3', '14', '1', '0', '1', '10.46']
    })
    st.dataframe(dd, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Preview Data (500 baris pertama)</div>',
                unsafe_allow_html=True)
    st.dataframe(df_f.head(500), use_container_width=True)

    st.markdown('<div class="section-title">Statistik Deskriptif</div>', unsafe_allow_html=True)
    st.dataframe(df_f[['Amount', 'amount_log', 'Month', 'Day']].describe().round(2),
                 use_container_width=True)