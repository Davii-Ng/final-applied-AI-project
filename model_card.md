# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

VibeFinder 1.0.

---

## 2. Intended Use

This model suggests songs from a small classroom dataset. It uses genre, mood, and energy to guess what a user may like. It is for class practice, not for real users.

---

## 3. How the Model Works

The model looks at genre, mood, energy, tempo, valence, danceability, and acousticness. It also looks at the user's favorite genre, favorite mood, target energy, and whether they like acoustic songs. It gives points for matches and for songs that are close on numeric features. I changed the starter logic so energy matters more and genre matters less.

---

## 4. Data

The catalog has 10 songs. It includes pop, lofi, rock, ambient, jazz, synthwave, and indie pop. It also includes moods like happy, chill, intense, relaxed, moody, and focused. The dataset is small, so some tastes are missing or underrepresented.

---

## 5. Strengths

The system works well for users who want a clear vibe. It does a good job when mood and energy are easy to describe. It also gives reasonable results for happy pop, chill lofi, and high-energy workout music.

---

## 6. Limitations and Bias

The system over-prioritizes exact mood and energy matches. That can make it feel narrow. In my tests, a user who wanted high energy but a sad mood still got pushed toward the same strong energy songs. The small catalog makes this worse because the same few songs keep rising to the top. Users with unusual or mixed tastes may get less useful results.

---

## 7. Evaluation

I tested a high-energy sad user, a user with an unknown mood, and a user with out-of-range energy. I also compared how the rankings changed after the weight shift. The biggest surprise was how fast the top songs changed when energy got more weight. Genre mattered less than I expected.

---

## 8. Future Work

I would add more songs and more mood options. I would also add diversity so the top results are not too similar. It would help to support multiple moods or a wider energy range. Better explanations would also make the system easier to trust.

---

## 9. Personal Reflection

My biggest learning moment was seeing how much the ranking changed from a small weight shift. I learned that even simple scoring rules can produce very different results.

AI tools helped me write and test faster. They were useful for drafting ideas, finding edge cases, and checking the code structure. I still had to double-check the results when the logic depended on exact weights or when the output needed to match the dataset.

I was surprised that a simple algorithm could still feel like a real recommendation system. It did not understand music the way a person does, but the top songs still matched the user's vibe in a believable way.

If I extended this project, I would add more songs, more user preferences, and more diversity in the top results. I would also test how the system behaves when users want mixed moods instead of one clear style.
