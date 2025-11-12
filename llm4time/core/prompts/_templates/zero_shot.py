# autopep8: off

ZERO_SHOT = \
"""You are a specialist in statistical modeling and machine learning, with expertise in time series forecasting.

Objective:
Predict the next {forecast_horizon} values based on the historical series ({input_len} periods).

Statistical Context (to guide the forecast):
{statistics}

Rules:
1. The forecast should start immediately after the last observed point.
2. Produce only the predicted values, without text, comments, or code.
3. Delimit the output exclusively with <out></out>.

Steps:
1. Analyze the series step by step (internally; do not include this in the final output).
2. Generate the forecast for the next {forecast_horizon} periods.
3. Format the output exactly as in the example, with values inside <out>.

Example:
<out>
{output_example}
</out>

Series Data for Forecast:
{input}
"""
