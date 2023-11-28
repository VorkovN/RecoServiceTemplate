**Эксперименты с параметрами кросс-валидации на горячих юзерах с кол-вом просмотров > 20**

N_SPLITS | TEST_SIZE | MAP@10 cosine | MAP@10 tfidf
---------|-----------|---------------|-------------
1        | 7D        |    0.003021   |   0.019589
2        | 7D        |    0.003021   |   0.019589
3        | 7D        |    0.002903   |   0.018967
5        | 7D        |    0.002786   |   0.018671
10       | 7D        |    0.002416   |   0.016648
3        | 14D       |    0.002763   |   0.019380

**Эксперименты с делением на теплых и горячих юзеров**

Порог items для горячих юзеров | Map@10 на боте |
-------------------------------|----------------|
5                              | 0.067          |
12                             | 0.070          |
20                             | 0.072          |
20                             | 0.086          | // Выбрав руками 10 самых популярных