"""Independent formulas for the migrated numerical workloads."""

import math
from contracts import require


def welch(x, y, method="welch"):
    import mpmath as mp

    require(mp.__version__ == "1.4.1", "Welch reference requires pinned mpmath 1.4.1")
    with mp.workdps(60):
        n, m = mp.mpf(len(x)), mp.mpf(len(y))
        mx, my = mp.fsum(x) / n, mp.fsum(y) / m
        vx = mp.fsum((mp.mpf(v) - mx) ** 2 for v in x) / (n - 1)
        vy = mp.fsum((mp.mpf(v) - my) ** 2 for v in y) / (m - 1)
        pooled = mp.sqrt(((n - 1) * vx + (m - 1) * vy) / (n + m - 2))
        difference = mx - my
        if method == "paired":
            delta = [mp.mpf(b) - mp.mpf(a) for a, b in zip(x, y)]
            difference = mp.fsum(delta) / n
            pooled = mp.sqrt(mp.fsum((v - difference) ** 2 for v in delta) / (n - 1))
            se2, df = pooled**2 / n, n - 1
        elif method == "student":
            se2, df = pooled**2 * (1 / n + 1 / m), n + m - 2
        else:
            se2 = vx / n + vy / m
            df = se2**2 / ((vx / n) ** 2 / (n - 1) + (vy / m) ** 2 / (m - 1))
        se = mp.sqrt(se2)
        t = difference / se
        factor = mp.exp(mp.loggamma((df + 1) / 2) - mp.loggamma(df / 2)) / mp.sqrt(
            df * mp.pi
        )

        def area(z):
            return mp.quad(
                lambda u: factor * (1 + u * u / df) ** (-(df + 1) / 2), [0, z]
            )

        p = 1 - 2 * area(abs(t))
        critical = mp.findroot(lambda z: area(z) - mp.mpf("0.475"), (1, 3))
        margin = critical * se
        return [
            float(v)
            for v in [
                t,
                p,
                df,
                difference - margin,
                difference + margin,
                difference / pooled,
            ]
        ]


def regression_metrics(x, y):
    errors = [a - b for a, b in zip(x, y)]
    mean = math.fsum(x) / len(x)
    ss = math.fsum(v * v for v in errors)
    valid = [abs((a - b) / a) for a, b in zip(x, y) if abs(a) > 1e-12]
    return [
        math.sqrt(ss / len(x)),
        math.fsum(abs(v) for v in errors) / len(x),
        100 * math.fsum(valid) / len(valid) if valid else 0,
        1 - ss / math.fsum((v - mean) ** 2 for v in x),
    ]


def auc(labels, scores):
    # Count pairwise wins using equal-score blocks, independently of ROC integration.
    ordered = sorted(zip(scores, labels))
    negatives = wins = 0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        positives = sum(label for _, label in ordered[start:end])
        ties = end - start - positives
        wins += positives * (negatives + ties / 2)
        negatives += ties
        start = end
    positives = sum(labels)
    return wins / (positives * negatives)


def tfidf(rows):
    # Alternating documents: "alpha beta beta" and "beta gamma".
    na, ng = (rows + 1) // 2, rows // 2
    ia, ig = math.log((rows + 1) / (na + 1)) + 1, math.log((rows + 1) / (ng + 1)) + 1
    return [
        v
        for i in range(rows)
        for v in ([ia / 3, 2 / 3, 0] if i % 2 == 0 else [0, 1 / 2, ig / 2])
    ]


def anova(groups):
    import mpmath as mp

    require(mp.__version__ == "1.4.1", "ANOVA reference requires pinned mpmath 1.4.1")
    with mp.workdps(60):
        count = sum(map(len, groups))
        means = [mp.fsum(g) / len(g) for g in groups]
        mean = mp.fsum(mp.fsum(g) for g in groups) / count
        between = mp.fsum(len(g) * (m - mean) ** 2 for g, m in zip(groups, means))
        within = mp.fsum((mp.mpf(v) - m) ** 2 for g, m in zip(groups, means) for v in g)
        db, dw = len(groups) - 1, count - len(groups)
        f = (between / db) / (within / dw)
        p = mp.betainc(
            mp.mpf(dw) / 2, mp.mpf(db) / 2, 0, dw / (dw + db * f), regularized=True
        )
        return list(map(float, [f, p, db, dw, between / (between + within)]))
