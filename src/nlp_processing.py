import spacy
from spacy import displacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def load_reviews_data(file_path='data/sample_reviews.csv'):
    """
    Load reviews data from CSV file
    """
    df = pd.read_csv(file_path)
    return df

def process_text_with_spacy(text, nlp):
    """
    Process text with spaCy NLP pipeline
    """
    return nlp(text)

def extract_entities(doc):
    """
    Extract named entities from spaCy document
    """
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def analyze_sentiment(text, positive_words, negative_words):
    """
    Simple rule-based sentiment analysis
    """
    positive_count = sum(1 for word in positive_words if word in text.lower())
    negative_count = sum(1 for word in negative_words if word in text.lower())
    
    if positive_count > negative_count:
        return "Positive", positive_count, negative_count
    elif negative_count > positive_count:
        return "Negative", positive_count, negative_count
    else:
        return "Neutral", positive_count, negative_count

def visualize_ner(doc):
    """
    Visualize named entities using spaCy's displacy
    """
    return displacy.render(doc, style="ent", jupyter=False)

def run_nlp_analysis():
    """
    Complete workflow for NLP analysis
    """
    # Load spaCy model
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    # Load reviews data
    print("Loading reviews data...")
    df = load_reviews_data()
    
    # Define sentiment words
    positive_words = ["amazing", "excellent", "great", "good", "fast", "comfortable", 
                     "stylish", "outstanding", "beautiful", "satisfied", "vibrant"]
    negative_words = ["terrible", "disappointed", "damaged", "torn", "bent", "poor",
                     "overheats", "jams", "expensive", "uncomfortable"]
    
    results = []
    
    print("Processing reviews...")
    for _, row in df.iterrows():
        review_text = row['review_text']
        doc = process_text_with_spacy(review_text, nlp)
        entities = extract_entities(doc)
        sentiment, pos_count, neg_count = analyze_sentiment(review_text, positive_words, negative_words)
        
        results.append({
            'review_id': row['review_id'],
            'product_id': row['product_id'],
            'review_text': review_text,
            'entities': entities,
            'sentiment': sentiment,
            'positive_words': pos_count,
            'negative_words': neg_count,
            'rating': row['rating']
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\nNLP Analysis Results:")
    print("=" * 50)
    for _, row in results_df.iterrows():
        print(f"Review {row['review_id']}:")
        print(f"  Text: {row['review_text'][:50]}...")
        print(f"  Entities: {row['entities']}")
        print(f"  Sentiment: {row['sentiment']} (Pos: {row['positive_words']}, Neg: {row['negative_words']})")
        print(f"  Rating: {row['rating']}/5")
        print()
    
    # Visualize entity distribution
    all_entities = [ent for sublist in results_df['entities'] for ent in sublist]
    entity_types = [ent[1] for ent in all_entities]
    
    entity_counts = Counter(entity_types)
    plt.figure(figsize=(10, 6))
    plt.bar(entity_counts.keys(), entity_counts.values())
    plt.title('Named Entity Distribution')
    plt.xlabel('Entity Types')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Visualize sentiment distribution
    sentiment_counts = results_df['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    plt.title('Sentiment Distribution')
    plt.show()
    
    return results_df, nlp

if __name__ == "__main__":
    run_nlp_analysis()