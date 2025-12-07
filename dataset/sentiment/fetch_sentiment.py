import requests
import time
import pandas as pd
from datetime import datetime
import os

token = "c0ff6a347278eafd82108cc525eb97825e4cc28e"
symbol = ['BTC', 'ETH', 'XRP']
filter_list = ['bearish', 'bullish', 'neutral', 'rising', 'hot']

def get_sentiment_from_api(post):
    """
    Extract sentiment from CryptoPanic API response.
    The API may provide sentiment in different fields.
    """
    # Check various possible fields for sentiment
    sentiment = post.get("sentiment", "")
    if sentiment:
        # Map API sentiment to our labels
        sentiment_lower = sentiment.lower()
        if sentiment_lower in ['positive', 'bullish', 'rising', 'hot']:
            return "positive"
        elif sentiment_lower in ['negative', 'bearish']:
            return "negative"
        else:
            return "neutral"
    
    # If no sentiment field, check votes or other indicators
    votes = post.get("votes", {})
    if votes:
        positive_votes = votes.get("positive", 0)
        negative_votes = votes.get("negative", 0)
        if positive_votes > negative_votes:
            return "positive"
        elif negative_votes > positive_votes:
            return "negative"
    
    # Default to neutral if no sentiment info available
    return "neutral"

def parse_published_at(published_at_str):
    """
    Parse published_at string to separate date and time.
    CryptoPanic format is typically: "2025-04-16T21:30:10Z" or similar ISO format
    """
    try:
        # Try parsing ISO format
        dt = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
        # Convert to local time if needed (remove timezone)
        dt = dt.replace(tzinfo=None)
        
        published_date = dt.strftime('%d/%m/%Y')
        published_time = dt.strftime('%H:%M:%S')
        
        return published_date, published_time
    except Exception as e:
        # Fallback parsing
        try:
            # Try other common formats
            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']:
                try:
                    dt = datetime.strptime(published_at_str.replace('Z', ''), fmt)
                    published_date = dt.strftime('%d/%m/%Y')
                    published_time = dt.strftime('%H:%M:%S')
                    return published_date, published_time
                except:
                    continue
        except:
            pass
        
        # If all parsing fails, return current date/time
        now = datetime.now()
        return now.strftime('%d/%m/%Y'), now.strftime('%H:%M:%S')

def fetch_cryptopanic_news(symbol, token, filter_list, existing_urls=None, target_count=1000):
    """
    Fetch news from CryptoPanic API for a specific symbol until reaching target count.
    
    Args:
        symbol: Cryptocurrency symbol (BTC, ETH, XRP)
        token: CryptoPanic API auth token
        filter_list: List of filters to apply
        existing_urls: Set of existing URLs to skip duplicates
        target_count: Target number of posts to fetch (default: 1000)
    
    Returns:
        List of dictionaries with post data (only new posts, no duplicates)
    """
    # Join filter list with commas for URL
    filter_str = ','.join(filter_list)
    base_url = f"https://cryptopanic.com/api/developer/v2/posts/?auth_token={token}&currencies={symbol}&filter={filter_str}"
    
    all_posts = []
    existing_urls = existing_urls or set()
    page = 1
    
    current_count = len(existing_urls)
    needed = target_count - current_count
    
    if needed <= 0:
        print(f"  ✅ Already have {current_count} posts, target of {target_count} reached!")
        return []
    
    print(f"Fetching {symbol} news from CryptoPanic...")
    print(f"  Current: {current_count} posts | Target: {target_count} | Need: {needed} more")
    
    while len(all_posts) < needed:
        try:
            # Add page parameter
            url = f"{base_url}&page={page}"
            print(f"  📄 Fetching page {page}...")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            
            if not results:
                print(f"  ⚠️  No more results on page {page}")
                break
            
            new_posts_this_page = 0
            
            # Process all posts (API already filters by currency)
            for post in results:
                title = post.get("title", "")
                published_at = post.get("published_at", "")
                post_id = post.get("id", "")
                
                # Get URL - CryptoPanic API may provide it in different fields
                url_field = post.get("url", "") or post.get("source", {}).get("url", "")
                
                # Skip if we already have this URL
                if url_field and url_field in existing_urls:
                    continue
                
                # Get sentiment from API response
                sentiment = get_sentiment_from_api(post)
                
                # Parse published_at
                published_date, published_time = parse_published_at(published_at)
                
                # Construct URL if not provided by API
                if not url_field:
                    # Create URL from post ID and title (matching CryptoPanic format)
                    if post_id:
                        # Clean title for URL slug - remove special chars, keep alphanumeric and hyphens
                        title_slug = title.replace("'", "").replace('"', '').replace('?', '').replace('!', '')
                        title_slug = title_slug.replace(',', '').replace('.', '').replace(':', '')
                        # Replace spaces and multiple hyphens with single hyphen
                        title_slug = '-'.join(title_slug.split())
                        # Remove any remaining special characters except hyphens
                        title_slug = ''.join(c for c in title_slug if c.isalnum() or c == '-')
                        # Remove multiple consecutive hyphens
                        while '--' in title_slug:
                            title_slug = title_slug.replace('--', '-')
                        url_field = f"https://cryptopanic.com/news/{post_id}/{title_slug}?mtm_campaign=API-OFA"
                    else:
                        url_field = f"https://cryptopanic.com/news/{post_id}/?mtm_campaign=API-OFA"
                
                # Skip if URL is duplicate (after construction)
                if url_field in existing_urls:
                    continue
                
                all_posts.append({
                    "post": title,
                    "sentiment": sentiment,
                    "url": url_field,
                    "published_date": published_date,
                    "published_time": published_time
                })
                
                existing_urls.add(url_field)
                new_posts_this_page += 1
                
                # Check if we've reached our target
                if len(all_posts) >= needed:
                    print(f"  ✅ Reached target! Got {len(all_posts)} new posts")
                    break
            
            print(f"  Page {page}: Found {len(results)} posts, {new_posts_this_page} new (Total new: {len(all_posts)})")
            
            # Check if we've reached our target
            if len(all_posts) >= needed:
                break
            
            # Check if there's a next page
            if not data.get("next"):
                print(f"  ⚠️  No more pages available")
                break
            
            page += 1
            # Rate limiting - wait 1 second between pages to avoid rate limits
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  Error fetching page {page}: {e}")
            print(f"  ⏳ Waiting 5 seconds before retry...")
            time.sleep(5)
            # Continue to next page instead of breaking
            page += 1
            continue
        except Exception as e:
            print(f"  ⚠️  Error processing page {page}: {e}")
            print(f"  ⏳ Waiting 2 seconds before continuing...")
            time.sleep(2)
            page += 1
            continue
    
    print(f"  ✅ Fetched {len(all_posts)} new {symbol} posts")
    return all_posts

def get_existing_posts(symbol, output_dir="dataset/sentiment"):
    """
    Load existing posts from CSV file.
    
    Args:
        symbol: Cryptocurrency symbol
        output_dir: Output directory path
    
    Returns:
        Tuple of (DataFrame, set of existing URLs)
    """
    required_columns = ['post', 'sentiment', 'url', 'published_date', 'published_time']
    file_path = os.path.join(output_dir, f"{symbol.lower()}_sentiment.csv")
    
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        # Ensure existing data has only required columns
        if all(col in df_existing.columns for col in required_columns):
            df_existing = df_existing[required_columns]
            existing_urls = set(df_existing['url'].astype(str))
            return df_existing, existing_urls
    
    return pd.DataFrame(columns=required_columns), set()

def save_to_csv(posts, symbol, output_dir="dataset/sentiment"):
    """
    Append new posts to existing CSV file.
    Only saves the 5 required columns: post, sentiment, url, published_date, published_time
    
    Args:
        posts: List of post dictionaries (new posts only, already filtered for duplicates)
        symbol: Cryptocurrency symbol
        output_dir: Output directory path
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not posts:
        print(f"  ⚠️  No new posts to save for {symbol}")
        return
    
    # Convert to DataFrame
    df_new = pd.DataFrame(posts)
    
    # Ensure only the 5 required columns
    required_columns = ['post', 'sentiment', 'url', 'published_date', 'published_time']
    df_new = df_new[required_columns]
    
    # File path
    file_path = os.path.join(output_dir, f"{symbol.lower()}_sentiment.csv")
    
    # Load existing data
    df_existing, _ = get_existing_posts(symbol, output_dir)
    
    if not df_existing.empty:
        print(f"  📂 Appending to existing file with {len(df_existing)} posts")
        # Combine existing and new data
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Remove any remaining duplicates (in case of edge cases)
        df_combined = df_combined.drop_duplicates(subset=['url'], keep='last')
        
        # Sort by published_date (newest first)
        df_combined['published_date_parsed'] = pd.to_datetime(
            df_combined['published_date'], 
            format='%d/%m/%Y', 
            errors='coerce'
        )
        df_combined = df_combined.sort_values('published_date_parsed', ascending=False)
        df_combined = df_combined.drop('published_date_parsed', axis=1)
        
        df_final = df_combined
    else:
        df_final = df_new
        print(f"  📝 Creating new file for {symbol}")
    
    # Ensure only required columns before saving
    df_final = df_final[required_columns]
    
    # Save to CSV with proper headers
    df_final.to_csv(file_path, index=False)
    print(f"  💾 Saved {len(df_final)} total posts to {file_path} ({len(df_new)} new)")

def main():
    """Main function to fetch news for all cryptocurrencies until reaching 1000 posts each"""
    print("\n" + "="*60)
    print("CRYPTONE - CryptoPanic News Fetcher")
    print("Target: 1000 posts per cryptocurrency")
    print("="*60)
    
    # Use the token from module level
    if not token:
        print("\n❌ ERROR: Auth token not found!")
        print("Please set 'token' in fetch_sentiment.py")
        return
    
    target_count = 1000
    
    for x in symbol:
        try:
            print(f"\n{'='*60}")
            print(f"Processing {x}...")
            print(f"{'='*60}")
            
            # Get existing posts and URLs
            df_existing, existing_urls = get_existing_posts(x)
            current_count = len(df_existing)
            
            print(f"  📊 Current count: {current_count} posts")
            
            if current_count >= target_count:
                print(f"  ✅ Already have {current_count} posts (target: {target_count})")
                print(f"  ⏭️  Skipping {x}")
                continue
            
            # Fetch news until we reach target
            posts = fetch_cryptopanic_news(x, token, filter_list, existing_urls, target_count)
            
            # Save to CSV (append new posts)
            if posts:
                save_to_csv(posts, x)
            else:
                print(f"  ⚠️  No new posts found for {x}")
            
            # Wait between symbols to avoid rate limits
            if x != symbol[-1]:  # Don't wait after last symbol
                print(f"\n⏳ Waiting 3 seconds before next symbol...")
                time.sleep(3)
                
        except Exception as e:
            print(f"\n❌ Error processing {x}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("✅ All done! News data saved to dataset/sentiment/")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
