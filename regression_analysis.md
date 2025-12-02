regression_analysis
================

### Multiple Linear Regression

Fit the model

``` r
mlr_fit =
  lm(mean_glth ~ mean_mhlth + mean_depression + mean_access +
       mean_smoking + mean_binge + mean_sleep + mean_lpa +
       mean_diabetes + mean_bphigh + EI_proportion + PO_proportion,
     data = regression_data)

summary(mlr_fit) |> 
  broom::tidy(conf.int = TRUE) |> 
  select(term, estimate, conf.low, conf.high, p.value) |> 
  knitr::kable(digits = 3)
```

| term            | estimate | conf.low | conf.high | p.value |
|:----------------|---------:|---------:|----------:|--------:|
| (Intercept)     |   -6.425 |  -10.977 |    -1.873 |   0.007 |
| mean_mhlth      |    0.558 |    0.263 |     0.854 |   0.000 |
| mean_depression |    0.008 |   -0.134 |     0.149 |   0.914 |
| mean_access     |    0.029 |   -0.041 |     0.099 |   0.413 |
| mean_smoking    |   -0.021 |   -0.172 |     0.130 |   0.780 |
| mean_binge      |    0.004 |   -0.101 |     0.109 |   0.939 |
| mean_sleep      |   -0.101 |   -0.225 |     0.024 |   0.110 |
| mean_lpa        |    0.224 |    0.080 |     0.369 |   0.003 |
| mean_diabetes   |    0.671 |    0.287 |     1.055 |   0.001 |
| mean_bphigh     |    0.033 |   -0.111 |     0.178 |   0.643 |
| EI_proportion   |   12.789 |    5.606 |    19.971 |   0.001 |
| PO_proportion   |   -2.130 |   -4.270 |     0.011 |   0.051 |

Check for key assumptions

``` r
p1 =
  regression_data |>
  modelr::add_residuals(mlr_fit) |>
  modelr::add_predictions(mlr_fit) |>
  ggplot(aes(x = pred, y = resid)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red") +
  labs(x = "Fitted values", y = "Residuals",
       title = "Residuals vs Fitted")

p2 =
  ggplot(mlr_fit, aes(sample = .resid)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "Normal Q-Q Plot of Residuals")
```

    ## Warning: `fortify(<lm>)` was deprecated in ggplot2 3.6.0.
    ## ℹ Please use `broom::augment(<lm>)` instead.
    ## ℹ The deprecated feature was likely used in the ggplot2 package.
    ##   Please report the issue at <https://github.com/tidyverse/ggplot2/issues>.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

``` r
p1 + p2
```

<img src="regression_analysis_files/figure-gfm/unnamed-chunk-5-1.png" width="90%" />

``` r
reduced_fit =
  lm(mean_glth ~ mean_mhlth + mean_depression +
       mean_smoking + mean_binge + mean_sleep + mean_lpa +
       mean_diabetes + mean_bphigh +
       EI_proportion + PO_proportion,
     data = regression_data)

full_fit = mlr_fit
anova(reduced_fit, full_fit)
```

    ## Analysis of Variance Table
    ## 
    ## Model 1: mean_glth ~ mean_mhlth + mean_depression + mean_smoking + mean_binge + 
    ##     mean_sleep + mean_lpa + mean_diabetes + mean_bphigh + EI_proportion + 
    ##     PO_proportion
    ## Model 2: mean_glth ~ mean_mhlth + mean_depression + mean_access + mean_smoking + 
    ##     mean_binge + mean_sleep + mean_lpa + mean_diabetes + mean_bphigh + 
    ##     EI_proportion + PO_proportion
    ##   Res.Df    RSS Df Sum of Sq      F Pr(>F)
    ## 1     39 19.571                           
    ## 2     38 19.224  1   0.34664 0.6852  0.413

The residuals appear randomly scattered around zero without a clear
pattern, suggesting that the linearity and constant variance assumptions
are reasonably satisfied.

Most points fall close to the reference line, indicating that the
residuals are approximately normally distributed. A few deviations at
the tails are expected with small samples (n = 50), but overall the
normality assumption is acceptable.
