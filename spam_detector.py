
#  The script will:
#   - Download a real spam dataset automatically
#   - Train a Naive Bayes classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import urllib.request
import io
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)

# ── Styling ──────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  📧  EMAIL SPAM DETECTOR — Starting...")
print("═"*55)

print("\n[1/5] Loading dataset...")

# Download the classic SMS Spam Collection dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

try:
    with urllib.request.urlopen(url, timeout=10) as response:
        zip_data = io.BytesIO(response.read())
    with zipfile.ZipFile(zip_data) as z:
        with z.open('SMSSpamCollection') as f:
            df = pd.read_csv(f, sep='\t', header=None, names=['label', 'text'])
    print("   ✓ Dataset downloaded from UCI repository")
except Exception:
    # Fallback: build a small but realistic dataset
    print("   ⚠  Could not reach internet. Using built-in sample data.")
    spam_samples = [
        "WINNER!! You have been selected for a $1,000 prize. Call now to claim!",
        "FREE entry to our weekly competition! Text WIN to 80086 now!",
        "Congratulations! You've won a luxury car. Confirm your details immediately.",
        "URGENT: Your account has been compromised. Click here to verify now.",
        "You have been pre-approved for a loan of $5000. No credit check needed!",
        "Hot singles in your area want to meet YOU tonight! Click here.",
        "Earn $500/day working from home! No experience needed. Sign up free.",
        "Your mobile number has won a £2000 prize. Call 09061743810 now!",
        "SIX chances to win CASH! From £100 to £20,000 txt> CSH11 Send to 87575",
        "Claim your FREE ringtone now! Just text MUSIC to 85069.",
        "IMPORTANT: You have a pending bank transfer. Confirm your details.",
        "You are a WINNER! Click the link to receive your Amazon gift card.",
        "Get cheap meds online! Viagra, Cialis no prescription needed.",
        "Make money fast! Join our MLM network and earn unlimited income.",
        "Your account will be suspended. Verify immediately via the link below.",
        "Congratulations! iPhone 15 winner. Reply with your address to claim.",
        "Call 0800 to get your FREE insurance quote now! Don't miss out!!!",
        "You have 1 unread message from a secret admirer. Click to reveal.",
        "LOAN approved! Get up to £10000 instantly. No questions asked.",
        "FREE! Free! FREE! Download our app and get 100 coins instantly!",
        "Act now! Limited offer expires tonight. Buy 1 get 2 FREE!!!",
        "Win a holiday! Just answer this simple question. Text HOLIDAY to 12345.",
        "Your delivery is on hold. Update your payment details immediately.",
        "Congratulations, you've been selected for our survey. Win £500 today!",
        "ALERT: Unusual sign-in detected. Verify your identity NOW.",
    ] * 8

    ham_samples = [
        "Hey, are you coming to the meeting tomorrow?",
        "Can you pick up some groceries on the way home?",
        "Happy birthday! Hope you have a wonderful day.",
        "I'll be there in 10 minutes. Just finishing up.",
        "Did you watch the game last night? Incredible ending!",
        "Can we reschedule our call to 3pm instead?",
        "Thanks for dinner, it was really lovely!",
        "The report is due Friday. Let me know if you need help.",
        "Mom called, she wants you to ring her back tonight.",
        "I left my umbrella at your place, can I grab it tomorrow?",
        "The project looks great! Nice work on the presentation.",
        "Do you want to grab coffee this afternoon?",
        "Sorry I missed your call. In a meeting. Will call back soon.",
        "The kids had a great time at the party, thank you!",
        "Just checking in. How are things going?",
        "Reminder: dentist appointment tomorrow at 2pm.",
        "Can you send me the address for Saturday?",
        "Flight is delayed by 2 hours. Will update you.",
        "Happy new year! Wishing you all the best.",
        "I finished the book you recommended. It was amazing!",
        "Don't forget to submit your timesheet by 5pm today.",
        "Are you free for lunch this week?",
        "Just landed safely. Talk soon!",
        "The wifi password is homesweet24, let me know if it works.",
        "Good luck on your exam! You've prepared really well.",
        "Running a bit late, start without me.",
        "Loved the photos from your trip! Looks beautiful.",
        "Can you cover my shift on Sunday? I'll owe you one.",
        "The meeting has been moved to the conference room B.",
        "Thanks for the recommendation, I'll check it out!",
    ] * 8

    texts  = spam_samples + ham_samples
    labels = ['spam'] * len(spam_samples) + ['ham'] * len(ham_samples)
    df = pd.DataFrame({'label': labels, 'text': texts})

print(f"   ✓ Total messages loaded: {len(df)}")
print(f"   ✓ Spam: {(df['label']=='spam').sum()}  |  Ham: {(df['label']=='ham').sum()}")


# ══════════════════════════════════════════════════════════════
#  STEP 2 — PREPROCESS
# ══════════════════════════════════════════════════════════════
print("\n[2/5] Preprocessing text...")

# Convert labels to numbers: spam=1, ham=0
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_num'],
    test_size=0.2,
    random_state=42,
    stratify=df['label_num']   # keeps spam/ham ratio balanced
)

# TF-IDF: converts raw text into numbers the model can understand.
# It counts word frequency but penalises very common words.
vectorizer = TfidfVectorizer(
    stop_words='english',   # removes "the", "is", "a", etc.
    max_features=5000,      # only keep top 5000 words
    ngram_range=(1, 2)      # uses single words AND pairs of words
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"   ✓ Training samples : {X_train_tfidf.shape[0]}")
print(f"   ✓ Test samples     : {X_test_tfidf.shape[0]}")
print(f"   ✓ Features (words) : {X_train_tfidf.shape[1]}")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════
print("\n[3/5] Training Naive Bayes classifier...")

# Multinomial Naive Bayes is the classic algorithm for text classification.
# It's fast, interpretable, and works really well for spam detection.
model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)

print("   ✓ Model trained successfully!")


# ══════════════════════════════════════════════════════════════
#  STEP 4 — EVALUATE
# ══════════════════════════════════════════════════════════════
print("\n[4/5] Evaluating model performance...")

y_pred      = model.predict(X_test_tfidf)
y_prob      = model.predict_proba(X_test_tfidf)[:, 1]

accuracy    = accuracy_score(y_test, y_pred)
precision   = precision_score(y_test, y_pred)
recall      = recall_score(y_test, y_pred)
f1          = f1_score(y_test, y_pred)
cm          = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)

print("\n" + "─"*40)
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")
print(f"  ROC AUC   : {roc_auc:.4f}")
print("─"*40)
print("\n  Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))


# ══════════════════════════════════════════════════════════════
#  STEP 5 — VISUALISE
# ══════════════════════════════════════════════════════════════
print("[5/5] Generating plots...")

fig = plt.figure(figsize=(18, 14), facecolor='#0f0f1a')
fig.suptitle(
    '📧  Email Spam Detector — Model Evaluation',
    fontsize=22, fontweight='bold', color='white', y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)


# ── Plot 1: Metrics Bar Chart ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values  = [accuracy, precision, recall, f1]
colors  = [ACCENT, SPAM_COLOR, HAM_COLOR, '#ffd32a']

bars = ax1.bar(metrics, [v * 100 for v in values], color=colors,
               width=0.55, edgecolor='none', zorder=3)

for bar, val in zip(bars, values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f'{val*100:.1f}%',
        ha='center', va='bottom', fontsize=11, fontweight='bold', color='white'
    )

ax1.set_ylim(0, 115)
ax1.set_title('Model Metrics', fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel('Score (%)')
ax1.tick_params(axis='x', labelsize=9)
ax1.grid(axis='y', zorder=0)
ax1.set_axisbelow(True)


# ── Plot 2: Confusion Matrix ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
labels_cm = ['Ham\n(Not Spam)', 'Spam']
sns.heatmap(
    cm, annot=True, fmt='d', ax=ax2,
    cmap=sns.color_palette("Blues", as_cmap=True),
    linewidths=2, linecolor='#0f0f1a',
    xticklabels=labels_cm, yticklabels=labels_cm,
    annot_kws={'size': 16, 'weight': 'bold'}
)
ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')


# ── Plot 3: ROC Curve ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(fpr, tpr, color=ACCENT, lw=2.5,
         label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='#555', lw=1.5, linestyle='--',
         label='Random Classifier')
ax3.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1.02])
ax3.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=12)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.legend(loc='lower right', fontsize=9, framealpha=0.2)
ax3.grid(True)


# ── Plot 4: Top Spam Words ────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
feature_names = vectorizer.get_feature_names_out()
spam_log_probs = model.feature_log_prob_[1]   # log prob for class=spam

# Get top 15 words most associated with spam
top_spam_idx   = spam_log_probs.argsort()[-15:][::-1]
top_spam_words = [feature_names[i] for i in top_spam_idx]
top_spam_probs = spam_log_probs[top_spam_idx]

# Normalise to 0–1 for display
top_spam_probs_norm = (top_spam_probs - top_spam_probs.min()) / \
                      (top_spam_probs.max() - top_spam_probs.min() + 1e-9)

colors_spam = [plt.cm.Reds(0.4 + 0.5 * v) for v in top_spam_probs_norm]
ax4.barh(top_spam_words[::-1], top_spam_probs_norm[::-1],
         color=colors_spam[::-1], edgecolor='none')
ax4.set_title('Top Spam Indicator Words', fontsize=14, fontweight='bold', pad=12)
ax4.set_xlabel('Relative Importance')
ax4.grid(axis='x')


# ── Plot 5: Top Ham Words ─────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ham_log_probs = model.feature_log_prob_[0]    # log prob for class=ham

top_ham_idx   = ham_log_probs.argsort()[-15:][::-1]
top_ham_words = [feature_names[i] for i in top_ham_idx]
top_ham_probs = ham_log_probs[top_ham_idx]

top_ham_probs_norm = (top_ham_probs - top_ham_probs.min()) / \
                     (top_ham_probs.max() - top_ham_probs.min() + 1e-9)

colors_ham = [plt.cm.Greens(0.4 + 0.5 * v) for v in top_ham_probs_norm]
ax5.barh(top_ham_words[::-1], top_ham_probs_norm[::-1],
         color=colors_ham[::-1], edgecolor='none')
ax5.set_title('Top Ham Indicator Words', fontsize=14, fontweight='bold', pad=12)
ax5.set_xlabel('Relative Importance')
ax5.grid(axis='x')


# ── Plot 6: Dataset Class Distribution ───────────────────────
ax6 = fig.add_subplot(gs[1, 2])
class_counts = df['label'].value_counts()
wedge_props  = {'linewidth': 3, 'edgecolor': '#0f0f1a'}

wedges, texts, autotexts = ax6.pie(
    class_counts,
    labels=['Ham (Safe)', 'Spam'],
    colors=[HAM_COLOR, SPAM_COLOR],
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=wedge_props,
    textprops={'color': 'white', 'fontsize': 11}
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')

ax6.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold', pad=12)


# ── Watermark / Footer ────────────────────────────────────────
fig.text(
    0.5, 0.01,
    'Model: Multinomial Naive Bayes  •  Features: TF-IDF (5000)  •  Train/Test: 80/20',
    ha='center', fontsize=9, color='#666'
)

plt.savefig('spam_detector_results.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
print("   ✓ Saved: spam_detector_results.png")
plt.show()


# ══════════════════════════════════════════════════════════════
#  BONUS — TRY YOUR OWN EMAILS
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  🔍  TRY YOUR OWN MESSAGES")
print("═"*55)

test_messages = [
    "Congratulations! You've won a free iPhone. Click here now!",
    "Hey, are we still meeting for coffee tomorrow at 10?",
    "URGENT: Your bank account has been compromised. Verify immediately.",
    "Can you review the report and send me your feedback?",
    "FREE offer! Get 50% off all items today only. Limited time!",
]

print()
for msg in test_messages:
    vec  = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]
    tag  = "🚨 SPAM" if pred == 1 else "✅ HAM "
    print(f"  {tag}  ({prob*100:.1f}% spam)  →  \"{msg[:60]}...\"" if len(msg) > 60
          else f"  {tag}  ({prob*100:.1f}% spam)  →  \"{msg}\"")

print("\n" + "═"*55)
print("  ✅  Done! Check 'spam_detector_results.png' for plots.")
print("═"*55 + "\n")