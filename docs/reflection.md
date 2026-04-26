# Reflection

## Conflict profile vs. unknown mood profile
The conflict profile, which asked for very high energy but a sad mood, kept pulling up songs like Gym Hero and Sunrise City because the system still gives a lot of credit to energy and exact mood matches. The unknown mood profile shifted toward lofi songs like Midnight Coding, Focus Flow, and Library Rain because the mood was not recognized, so the recommender fell back to more general audio features instead of a specific mood target. That makes sense because the first profile sends mixed signals, while the second one is treated more like a broad, calm listening request.

## Conflict profile vs. out-of-range energy profile
The out-of-range energy profile pushed songs like Sunrise City and Rooftop Lights to the top because the energy score stops being useful when the target is 1.8, so mood and genre matter more. The conflict profile still ranked Gym Hero highly, but it also let Storm Runner rise because the strong energy preference was fighting with the sad mood. This explains why Gym Hero keeps showing up for people who want "Happy Pop": it gets both the pop and happy bonuses, and its high energy also matches the system's preferred high-energy pattern.

## Unknown mood profile vs. out-of-range energy profile
The unknown mood profile stayed in the lofi, chill, and acoustic lane because the system did not know how to treat "bittersweet," so it used a generic fallback mood target. The out-of-range energy profile moved toward brighter pop songs instead, even though the energy number itself was unrealistic, because the model still rewards mood and genre matches more than it rewards a bad energy value. In plain language, one profile sounds like "calm and undefined," while the other sounds like "happy pop but with a broken energy slider."
