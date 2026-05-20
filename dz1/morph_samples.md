# Примеры преобразований

## Стемминг (Porter)

### Пример 1
Исходные токены:
`these institutions are often described as stateless societies although several authors have defined them more specifically as distinct institutions based`

После преобразования:
`these institut are often describ as stateless societi although sever author have defin them more specif as distinct institut base`

### Пример 2
Исходные токены:
`parents usually notice signs during the first three years of their child s life these signs often develop gradually though`

После преобразования:
`parent usual notic sign dure the first three year of their child s life these sign often develop gradual though`

### Пример 3
Исходные токены:
`a planet like earth it is dimensionless and measured via an albedometer on a scale from 0 corresponding to a`

После преобразования:
`a planet like earth it is dimensionless and measur via an albedomet on a scale from 0 correspond to a`

## Лемматизация (spaCy)

### Пример 1
Исходные токены:
`these institutions are often described as stateless societies although several authors have defined them more specifically as distinct institutions based`

После преобразования:
`these institution be often describe as stateless society although several author have define they more specifically as distinct institution base`

### Пример 2
Исходные токены:
`parents usually notice signs during the first three years of their child s life these signs often develop gradually though`

После преобразования:
`parent usually notice sign during the first three year of their child s life these sign often develop gradually though`

### Пример 3
Исходные токены:
`a planet like earth it is dimensionless and measured via an albedometer on a scale from 0 corresponding to a`

После преобразования:
`a planet like earth it be dimensionless and measure via an albedometer on a scale from 0 correspond to a`

## BERT-токенизация

### Пример 1
Исходные токены:
`these institutions are often described as stateless societies although several authors have defined them more specifically as distinct institutions based`

После преобразования:
`these institutions are often described as state ##less societies although several authors have defined them more specifically as distinct institutions`

### Пример 2
Исходные токены:
`parents usually notice signs during the first three years of their child s life these signs often develop gradually though`

После преобразования:
`parents usually notice signs during the first three years of their child s life these signs often develop gradually though`

### Пример 3
Исходные токены:
`a planet like earth it is dimensionless and measured via an albedometer on a scale from 0 corresponding to a`

После преобразования:
`a planet like earth it is dimension ##less and measured via an al ##bedo ##meter on a scale from 0`

