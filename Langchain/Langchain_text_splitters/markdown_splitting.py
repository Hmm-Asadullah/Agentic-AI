from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
# Hangman Game in C++

Welcome to the Hangman Game, a classic word-guessing game implemented in C++. This project includes user registration, login functionality, and the ability to save game progress. The game randomly selects words, and players try to guess the word one letter at a time.

## Features

- **User Management**:
  - Register new users with unique usernames and passwords.
  - Login functionality to access the game.
  
- **Gameplay**:
  - Random word selection from a predefined word list.
  - ASCII art visualization of the Hangman as incorrect guesses are made.
  - Tracks user performance (win/loss records and scores).

- **Persistent Records**:
  - Stores user credentials securely in a file.
  - Saves game records (win/loss and scores) for each user.

## File Structure

- `users.txt`: Stores registered usernames and passwords.
- `words.txt`: Contains a list of possible words for the Hangman game.
- `game_records.txt`: Logs game results for each user.

## How to Run

1. **Compile the Code**:
   Use a C++ compiler (e.g., g++) to compile the program:
   ```bash
   g++ -o hangman hangman.cpp
   ```

2. **Prepare Files**:
   - Create the following files in the directory specified in the code:
     - `users.txt` (for storing user credentials)
     - `words.txt` (populate this file with words, one word per line)
     - `game_records.txt` (to store game results)

3. **Run the Program**:
   Execute the compiled program:
   ```bash
   ./hangman
   ```

4. **Play the Game**:
   - Register or log in to start playing.
   - Guess letters to find the hidden word.
   - Win by guessing the word before the Hangman is fully drawn!


## Future Improvements

- Add a GUI version for a better user experience.
- Enhance word selection with categories.
- Implement leaderboard functionality.

## Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request.

## License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute the code as per the license terms.

Enjoy the game!
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=500,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])