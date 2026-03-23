import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0f0f1a; color: #e0e0e0; }
    .header-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #a29bfe;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .header-box h1 { color: #a29bfe; margin: 0; font-size: 32px; }
    .header-box p  { color: #aaa; margin: 6px 0 0; font-size: 14px; }
    .result-spam {
        background: #2d0a0a;
        border: 2px solid #ff4757;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-ham {
        background: #0a2d0a;
        border: 2px solid #2ed573;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-spam h2 { color: #ff4757; font-size: 36px; margin: 0; }
    .result-ham  h2 { color: #2ed573; font-size: 36px; margin: 0; }
    .result-spam p  { color: #ff6b81; margin: 6px 0 0; }
    .result-ham  p  { color: #7bed9f; margin: 6px 0 0; }
    </style>
""", unsafe_allow_html=True)

# ── Plot styling ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor':   '#1a1a2e',
    'axes.edgecolor':   '#444',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#aaa',
    'ytick.color':      '#aaa',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2a3e',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
})

SPAM_COLOR = '#ff4757'
HAM_COLOR  = '#2ed573'
ACCENT     = '#a29bfe'


# ── Step 1: Define functions ─────────────────────────────────
# functions must be defined BEFORE they are called

@st.cache_data
def load_data():
    df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'text'])
    return df, "local file"

@st.cache_resource
def train_model(df):
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label_num'],
        test_size=0.2, random_state=42,
        stratify=df['label_num']
    )
    vectorizer = TfidfVectorizer(
        stop_words='english', max_features=5000, ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)

    y_pred      = model.predict(X_test_tfidf)
    y_prob      = model.predict_proba(X_test_tfidf)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   auc(fpr, tpr),
        'cm':        confusion_matrix(y_test, y_pred),
        'fpr':       fpr,
        'tpr':       tpr,
    }
    return model, vectorizer, metrics


# ── Step 2: Call functions ───────────────────────────────────
# now that functions are defined above, we can call them here

df, source = load_data()
model, vectorizer, metrics = train_model(df)


# ── Step 3: Sidebar ──────────────────────────────────────────
# sidebar code comes AFTER load_data() so df and source exist

st.sidebar.header("Quick Test Messages")
st.sidebar.write("Click any to auto-fill the checker above")
st.sidebar.markdown("---")
st.sidebar.caption("Dataset stats")
st.sidebar.write(f"Dataset source: {source}")
st.sidebar.write(f"Total messages: {len(df)}")
st.sidebar.write(f"Spam: {(df['label']=='spam').sum()}")
st.sidebar.write(f"Ham:  {(df['label']=='ham').sum()}")


# ── Step 4: Header ───────────────────────────────────────────
st.markdown(f"""
    <div class="header-box">
        <h1>📧 Email Spam Detector</h1>
        <p>Multinomial Naive Bayes · TF-IDF (5000 features) · Dataset: {source}</p>
    </div>
""", unsafe_allow_html=True)


# ── Step 5: Metric cards ─────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy",  f"{metrics['accuracy']*100:.1f}%")
c2.metric("Precision", f"{metrics['precision']*100:.1f}%")
c3.metric("Recall",    f"{metrics['recall']*100:.1f}%")
c4.metric("F1 Score",  f"{metrics['f1']*100:.1f}%")
c5.metric("ROC AUC",   f"{metrics['roc_auc']:.3f}")

st.markdown("---")


# ── Step 6: Live spam checker ────────────────────────────────
st.subheader("Try it — check any message")

user_input = st.text_area(
    "Paste or type a message below",
    placeholder="e.g. Congratulations! You've won a free iPhone. Click here now!",
    height=100
)

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        vec  = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]

        if pred == 1:
            st.markdown(f"""
                <div class="result-spam">
                    <h2>🚨 SPAM</h2>
                    <p>This message is {prob*100:.1f}% likely to be spam</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-ham">
                    <h2>✅ HAM (Safe)</h2>
                    <p>This message is {(1-prob)*100:.1f}% likely to be safe</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")


# ── Step 7: Charts ───────────────────────────────────────────
st.subheader("Model Evaluation Charts")

fig = plt.figure(figsize=(18, 14), facecolor='#0f0f1a')
fig.suptitle(
    '📧  Email Spam Detector — Model Evaluation',
    fontsize=22, fontweight='bold', color='white', y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Metrics Bar
ax1 = fig.add_subplot(gs[0, 0])
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values  = [metrics['accuracy'], metrics['precision'],
           metrics['recall'], metrics['f1']]
colors  = [ACCENT, SPAM_COLOR, HAM_COLOR, '#ffd32a']
bars = ax1.bar(metric_names, [v * 100 for v in values],
               color=colors, width=0.55, edgecolor='none', zorder=3)
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
             f'{val*100:.1f}%', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color='white')
ax1.set_ylim(0, 115)
ax1.set_title('Model Metrics', fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel('Score (%)')
ax1.tick_params(axis='x', labelsize=9)
ax1.grid(axis='y', zorder=0)

# Plot 2: Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(metrics['cm'], annot=True, fmt='d', ax=ax2,
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=2, linecolor='#0f0f1a',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
            annot_kws={'size': 16, 'weight': 'bold'})
ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

# Plot 3: ROC Curve
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(metrics['fpr'], metrics['tpr'], color=ACCENT, lw=2.5,
         label=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})")
ax3.plot([0,1],[0,1], color='#555', lw=1.5, linestyle='--',
         label='Random Classifier')
ax3.fill_between(metrics['fpr'], metrics['tpr'], alpha=0.15, color=ACCENT)
ax3.set_xlim([0,1]); ax3.set_ylim([0,1.02])
ax3.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=12)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.legend(loc='lower right', fontsize=9, framealpha=0.2)
ax3.grid(True)

# Plot 4: Top Spam Words
ax4 = fig.add_subplot(gs[1, 0])
feature_names  = vectorizer.get_feature_names_out()
spam_log_probs = model.feature_log_prob_[1]
top_spam_idx   = spam_log_probs.argsort()[-15:][::-1]
top_spam_words = [feature_names[i] for i in top_spam_idx]
top_spam_probs = spam_log_probs[top_spam_idx]
top_spam_norm  = (top_spam_probs - top_spam_probs.min()) / \
                 (top_spam_probs.max() - top_spam_probs.min() + 1e-9)
colors_spam = [plt.cm.Reds(0.4 + 0.5*v) for v in top_spam_norm]
ax4.barh(top_spam_words[::-1], top_spam_norm[::-1],
         color=colors_spam[::-1], edgecolor='none')
ax4.set_title('Top Spam Words', fontsize=14, fontweight='bold', pad=12)
ax4.set_xlabel('Relative Importance')
ax4.grid(axis='x')

# Plot 5: Top Ham Words
ax5 = fig.add_subplot(gs[1, 1])
ham_log_probs = model.feature_log_prob_[0]
top_ham_idx   = ham_log_probs.argsort()[-15:][::-1]
top_ham_words = [feature_names[i] for i in top_ham_idx]
top_ham_probs = ham_log_probs[top_ham_idx]
top_ham_norm  = (top_ham_probs - top_ham_probs.min()) / \
                (top_ham_probs.max() - top_ham_probs.min() + 1e-9)
colors_ham = [plt.cm.Greens(0.4 + 0.5*v) for v in top_ham_norm]
ax5.barh(top_ham_words[::-1], top_ham_norm[::-1],
         color=colors_ham[::-1], edgecolor='none')
ax5.set_title('Top Ham Words', fontsize=14, fontweight='bold', pad=12)
ax5.set_xlabel('Relative Importance')
ax5.grid(axis='x')

# Plot 6: Class Distribution
ax6 = fig.add_subplot(gs[1, 2])
class_counts = df['label'].value_counts()
ax6.pie(class_counts, labels=['Ham (Safe)', 'Spam'],
        colors=[HAM_COLOR, SPAM_COLOR], autopct='%1.1f%%',
        startangle=90, wedgeprops={'linewidth':3,'edgecolor':'#0f0f1a'},
        textprops={'color':'white','fontsize':11})
ax6.set_title('Class Distribution', fontsize=14, fontweight='bold', pad=12)

st.pyplot(fig)
