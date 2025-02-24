import sqlite3
from googleapiclient.discovery import build
from tqdm import tqdm

def initialize_db(db_name="youtube_comments.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,  
            comment_id TEXT UNIQUE,                
            author TEXT,
            comment TEXT,
            author_profile_image_url TEXT,
            author_channel_url TEXT,
            author_channel_id TEXT,
            published_at TEXT,
            updated_at TEXT,
            like_count INTEGER,
            viewer_rating TEXT,
            can_reply INTEGER,
            total_reply_count INTEGER,
            is_public INTEGER,
            valence TEXT,
            video_id TEXT,
            channel_id TEXT
        )
    ''')

    conn.commit()
    return conn, cursor

def insert_comment(cursor, comment_data):
    try:
        cursor.execute('''
            INSERT INTO comments (
                comment_id, author, comment, author_profile_image_url, author_channel_url, 
                author_channel_id, published_at, updated_at, like_count, viewer_rating, 
                can_reply, total_reply_count, is_public, valence, video_id, channel_id
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            comment_data['comment_id'], comment_data['author'], comment_data['comment'],
            comment_data['author_profile_image_url'], comment_data['author_channel_url'],
            comment_data['author_channel_id'], comment_data['published_at'], comment_data['updated_at'],
            comment_data['like_count'], comment_data['viewer_rating'], comment_data['can_reply'],
            comment_data['total_reply_count'], comment_data['is_public'], comment_data['valence'],
            comment_data['video_id'], comment_data['channel_id']
        ))
    except sqlite3.IntegrityError:
        print(f"Duplicate comment found: {comment_data['comment_id']}")


def get_comments(api_key, video_id, channel_id, valence, db_conn, db_cursor):
    youtube = build('youtube', 'v3', developerKey=api_key)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,  
        textFormat="plainText"
    )

    response = request.execute()
    total_comments = int(response["pageInfo"]["totalResults"])  

    # Initialize progress bar
    with tqdm(total=total_comments, desc="Downloading comments", unit="comment") as pbar:
        while request:
            response = request.execute()

            for item in response['items']:
                comment_id = item['snippet']['topLevelComment']['id']
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                author_profile_image_url = item['snippet']['topLevelComment']['snippet']['authorProfileImageUrl']
                author_channel_url = item['snippet']['topLevelComment']['snippet']['authorChannelUrl']
                author_channel_id = item['snippet']['topLevelComment']['snippet']['authorChannelId']['value']
                published_at = item['snippet']['topLevelComment']['snippet']['publishedAt']
                updated_at = item['snippet']['topLevelComment']['snippet']['updatedAt']
                like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
                viewer_rating = item['snippet']['topLevelComment']['snippet']['viewerRating']
                can_reply = item['snippet']['canReply']
                total_reply_count = item['snippet']['totalReplyCount']
                is_public = item['snippet']['isPublic']

                comment_data = {
                    'comment_id': comment_id,
                    'author': author,
                    'comment': comment,
                    'author_profile_image_url': author_profile_image_url,
                    'author_channel_url': author_channel_url,
                    'author_channel_id': author_channel_id,
                    'published_at': published_at,
                    'updated_at': updated_at,
                    'like_count': like_count,
                    'viewer_rating': viewer_rating,
                    'can_reply': can_reply,
                    'total_reply_count': total_reply_count,
                    'is_public': is_public,
                    'valence': valence,
                    'video_id': video_id,
                    'channel_id': channel_id
                }

                insert_comment(db_cursor, comment_data)
                pbar.update(1)

            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=100,
                    textFormat="plainText"
                )
            else:
                request = None

    db_conn.commit()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python script.py VALENCE API_KEY VIDEO_ID CHANNEL_ID")
        sys.exit(1)

    VALENCE = sys.argv[1]  # High or Low
    API_KEY = sys.argv[2]  # Your YouTube API key
    VIDEO_ID = sys.argv[3]  # The YouTube video ID
    CHANNEL_ID = sys.argv[4]  # The YouTube channel ID

    db_conn, db_cursor = initialize_db()

    get_comments(API_KEY, VIDEO_ID, CHANNEL_ID, VALENCE, db_conn, db_cursor)

    db_conn.close()
    print(f"Comments for video {VIDEO_ID} stored successfully.")