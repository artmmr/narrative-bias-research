import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from scipy import stats
from scipy.stats import kruskal, f_oneway
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class NarrativeAnalyzer:
    """Analyze narratives across different reality frames."""
    
    def __init__(self, jsonl_path):
        """Load data from JSONL file."""
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.df = pd.DataFrame(self.data)
        
    def compute_sentiment_polarity(self, text):
        """Compute sentiment polarity using TextBlob (-1 to 1)."""
        return TextBlob(text).sentiment.polarity
    
    def compute_sentiment_subjectivity(self, text):
        """Compute sentiment subjectivity using TextBlob (0 to 1)."""
        return TextBlob(text).sentiment.subjectivity
    
    def compute_lexical_diversity(self, text):
        """
        Compute Type-Token Ratio (TTR) as measure of lexical diversity.
        TTR = unique words / total words
        """
        tokens = word_tokenize(text.lower())
        if len(tokens) == 0:
            return 0
        return len(set(tokens)) / len(tokens)
    
    def compute_narrative_complexity(self, text):
        """
        Compute narrative complexity as average sentence length.
        Longer sentences often indicate more complex syntax.
        """
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return 0
        words = word_tokenize(text)
        return len(words) / len(sentences)
    
    def compute_causal_coherence(self, text):
        """
        Estimate causal coherence by counting causal connectives.
        Higher count suggests more explicit causal structure.
        """
        causal_markers = [
            'because', 'since', 'therefore', 'thus', 'hence', 'consequently',
            'as a result', 'due to', 'caused by', 'leads to', 'results in',
            'so', 'then', 'if', 'when', 'after', 'before'
        ]
        text_lower = text.lower()
        count = sum(text_lower.count(marker) for marker in causal_markers)
        # Normalize by text length (per 100 words)
        words = word_tokenize(text)
        if len(words) == 0:
            return 0
        return (count / len(words)) * 100
    
    def add_linguistic_features(self):
        """Add all linguistic metrics to the dataframe."""
        print("Computing linguistic features...")
        
        self.df['sentiment_polarity'] = self.df['output'].apply(self.compute_sentiment_polarity)
        self.df['sentiment_subjectivity'] = self.df['output'].apply(self.compute_sentiment_subjectivity)
        self.df['lexical_diversity'] = self.df['output'].apply(self.compute_lexical_diversity)
        self.df['narrative_complexity'] = self.df['output'].apply(self.compute_narrative_complexity)
        self.df['causal_coherence'] = self.df['output'].apply(self.compute_causal_coherence)
        self.df['narrative_length'] = self.df['word_count']
        
        print("✓ Linguistic features computed")
        
    def descriptive_statistics(self):
        """Generate descriptive statistics by reality frame."""
        metrics = [
            'sentiment_polarity', 'sentiment_subjectivity', 'lexical_diversity',
            'narrative_complexity', 'causal_coherence', 'narrative_length'
        ]
        
        print("\n" + "="*80)
        print("DESCRIPTIVE STATISTICS BY REALITY FRAME")
        print("="*80 + "\n")
        
        for metric in metrics:
            print(f"\n{metric.upper().replace('_', ' ')}")
            print("-" * 80)
            stats_table = self.df.groupby('frame')[metric].agg(['mean', 'std', 'min', 'max', 'median'])
            print(stats_table.round(4))
            
        return self.df.groupby('frame')[metrics].agg(['mean', 'std'])
    
    def test_normality(self, metric):
        """Test normality assumption using Shapiro-Wilk test."""
        frames = self.df['frame'].unique()
        normal = True
        
        for frame in frames:
            data = self.df[self.df['frame'] == frame][metric].dropna()
            if len(data) >= 3:  # Shapiro test requires at least 3 samples
                stat, p = stats.shapiro(data)
                if p < 0.05:
                    normal = False
                    break
        return normal
    
    def comparative_statistics(self):
        """Perform ANOVA or Kruskal-Wallis tests for each metric."""
        metrics = [
            'sentiment_polarity', 'sentiment_subjectivity', 'lexical_diversity',
            'narrative_complexity', 'causal_coherence', 'narrative_length'
        ]
        
        results = []
        
        print("\n" + "="*80)
        print("COMPARATIVE STATISTICAL TESTS")
        print("="*80 + "\n")
        
        for metric in metrics:
            # Prepare data groups by frame
            groups = [group[metric].dropna().values for name, group in self.df.groupby('frame')]
            
            # Test normality
            is_normal = self.test_normality(metric)
            
            # Choose appropriate test
            if is_normal:
                # ANOVA for normally distributed data
                stat, p_value = f_oneway(*groups)
                test_name = "ANOVA"
            else:
                # Kruskal-Wallis for non-normal data
                stat, p_value = kruskal(*groups)
                test_name = "Kruskal-Wallis"
            
            # Determine significance
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            results.append({
                'Metric': metric.replace('_', ' ').title(),
                'Test': test_name,
                'Statistic': stat,
                'p-value': p_value,
                'Significance': significance
            })
            
            print(f"{metric.replace('_', ' ').title()}")
            print(f"  Test: {test_name}")
            print(f"  Statistic: {stat:.4f}")
            print(f"  p-value: {p_value:.4f} {significance}")
            print(f"  Interpretation: {'Significant difference' if p_value < 0.05 else 'No significant difference'} between frames\n")
        
        return pd.DataFrame(results)
    
    def post_hoc_analysis(self, metric):
        """
        Perform pairwise comparisons between frames for a given metric.
        Uses Mann-Whitney U test with Bonferroni correction.
        """
        from itertools import combinations
        
        frames = sorted(self.df['frame'].unique())
        pairs = list(combinations(frames, 2))
        
        # Bonferroni correction
        alpha = 0.05 / len(pairs)
        
        print(f"\n" + "="*80)
        print(f"POST-HOC PAIRWISE COMPARISONS: {metric.replace('_', ' ').title()}")
        print(f"Bonferroni-corrected α = {alpha:.4f}")
        print("="*80 + "\n")
        
        results = []
        for frame1, frame2 in pairs:
            data1 = self.df[self.df['frame'] == frame1][metric].dropna()
            data2 = self.df[self.df['frame'] == frame2][metric].dropna()
            
            stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            significant = "Yes" if p_value < alpha else "No"
            
            results.append({
                'Comparison': f"{frame1} vs {frame2}",
                'U-statistic': stat,
                'p-value': p_value,
                'Significant': significant
            })
            
            print(f"{frame1} vs {frame2}")
            print(f"  U-statistic: {stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {significant}\n")
        
        return pd.DataFrame(results)
    
    def visualize_metrics(self, save_path=None):
        """Create visualization of all metrics across frames."""
        metrics = [
            'sentiment_polarity', 'sentiment_subjectivity', 'lexical_diversity',
            'narrative_complexity', 'causal_coherence', 'narrative_length'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Box plot
            self.df.boxplot(column=metric, by='frame', ax=ax)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Reality Frame')
            ax.set_ylabel(metric.replace('_', ' ').title())
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_path='analysis_report.txt'):
        """Generate a comprehensive text report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("NARRATIVE BIAS ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*80 + "\n")
            f.write(f"Total narratives: {len(self.df)}\n")
            f.write(f"Reality frames: {', '.join(self.df['frame'].unique())}\n")
            f.write(f"Narratives per frame:\n")
            for frame, count in self.df['frame'].value_counts().items():
                f.write(f"  {frame}: {count}\n")
            f.write("\n")
            
            # Descriptive statistics
            f.write("\n" + "="*80 + "\n")
            f.write("DESCRIPTIVE STATISTICS\n")
            f.write("="*80 + "\n")
            desc_stats = self.descriptive_statistics()
            f.write(str(desc_stats))
            
        print(f"\n✓ Report saved to {output_path}")


def analyze_narratives(jsonl_path='outputs.jsonl'):
    """Run complete analysis pipeline."""
    
    print("Starting Narrative Analysis...")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = NarrativeAnalyzer(jsonl_path)
    
    # Add linguistic features
    analyzer.add_linguistic_features()
    
    # Descriptive statistics
    desc_stats = analyzer.descriptive_statistics()
    
    # Comparative statistics
    comp_stats = analyzer.comparative_statistics()
    
    # Save comparative statistics
    comp_stats.to_csv('comparative_statistics.csv', index=False)
    print("\n✓ Comparative statistics saved to comparative_statistics.csv")
    
    # Post-hoc analysis for significant metrics
    print("\nPerforming post-hoc analyses...")
    for metric in ['sentiment_polarity', 'lexical_diversity', 'causal_coherence']:
        analyzer.post_hoc_analysis(metric)
    
    # Visualizations
    analyzer.visualize_metrics(save_path='metrics_visualization.png')
    
    # Save processed data
    analyzer.df.to_csv('narratives_with_metrics.csv', index=False)
    print("✓ Processed data saved to narratives_with_metrics.csv")
    
    # Generate report
    analyzer.generate_report()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return analyzer

analyzer = analyze_narratives('outputs.jsonl')
