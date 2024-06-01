# Description of submission

- Logistic regression
- Manually selected variables based on existing literature, removed when little to no effect

Variables included:
- Age (+ squared & cubic term): young people are unlikely to have found a partner and be in a situation to have kids, biology works against those who are older.
- Cohabitation duration (+ squared & cubic term): those who have been together for a longer period of time have made more investments, and are likely to make a next investment.
- Having a partner: having a partner increases the likelihood of having children, as there will be someone to raise your kids with.
- Wanting a child (absolute, in 1, 2 or 3 years): committing to wanting to have a child (soon) means the person will change their behavior, making it more likely to have a chlid.
- Being in church on a weekly basis: those who follow religious norms are more likely to reproduce.
- Being unable to make ends meet: those who really struggle financially do not have the money to think about having children.


I have also looked at other variables, including whether a person already had a child, owned a house, was in good health, wasn't obese, was not in a relationship in which the female partner worked, lived in a non-polluted area and did no care work for other family members, but these factors only seemed to increase the complexity of the model.

I have looked into machine learning techniques, but decided against using those, as I feel the added complexity won't add much here, and I do value the model to be well interpretable as well.