# Define prompts
prompt_music = """
Please write an short and objective one-sentence description based on the below musical commentary. Please use simple words and do not mention the quality aspects of the music.

{s}
""".strip()

def create_prompt(comment: str):
    return [
        {'role': 'system', 'content': 'You are a professional musician asked to review music-related comment.'},
        # Give example
        {'role': 'user', 'content': prompt_music.replace('{s}', "This song features a rubber instrument being played. The strumming is fast. The melody is played on one fretted string and other open strings. The melody is played on the lower octave and is later repeated on the higher octave. This song can be played at a folk party. This song has low recording quality, as if recorded from a mobile phone.")},
        {'role': 'assistant', 'content': "Folk tune played on a rubber instrument with quick strumming and a two-octave melody"},
        # Ask for response
        {'role': 'user', 'content': prompt_music.replace('{s}', comment)}
    ]
