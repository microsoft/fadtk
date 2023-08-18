# Define prompts
prompt_music = """
Musicality or "musical quality" is defined as the artistic and aesthetic content of the music (e.g. whether the melody and dynamics are expressed/performed well, or whether the voice is clear), while "acoustic quality" defines the quality of the sound recording (e.g. whether the sound is clean from recording noise).

Does the following music comment describe the music as high or low in musicality or acoustic quality? Please answer each with "High", "Medium", "Low", or "Not mentioned". No explanation is needed.

{s}
""".strip()

example_comment = "A female Arabic singer sings this beautiful melody with backup singers in vocal harmony. The song is medium tempo with a string section, Arabic percussion instruments, tambourine percussion, steady drum rhythm, groovy bass line and keyboard accompaniment. The song is romantic and celebratory in nature. The audio quality is very poor."

example_response = """
Musicality: High
Acoustic: Low
""".strip()

def create_prompt(comment: str):
    return [
        {'role': 'system', 'content': 'You are a professional musician asked to review music-related comment.'},
        # Give example
        {'role': 'user', 'content': prompt_music.replace('{s}', example_comment)},
        {'role': 'assistant', 'content': example_response},
        # Ask for response
        {'role': 'user', 'content': comment},
    ]