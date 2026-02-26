import plotly.graph_objects as go


def generate_grad_flow_interactive_html(
    layers: list,
    avg_grads: list,
    max_grads: list,
    l2_norms: list,
) -> str:
    """
    Creates an interactive Plotly bar chart of gradient flow through different layers,
    and returns the complete HTML as a string.

    In addition to mean and max gradients, this function computes the L2 norm of the gradients,
    to help assess an appropriate threshold for gradient norm clipping.

    The HTML can be logged to Comet ML using exp.log_html(html_str). The plot supports
    zooming, mouse hover for value inspection, and extra y-axis ticks.

    This function does not display the plot on screen.

    Parameters:
        named_parameters: Iterator of (name, parameter) pairs.

    Returns:
        A string containing the full HTML representation of the interactive plot.
    """
    trace_max = go.Bar(
        x=layers,
        y=max_grads,
        name="Max Gradient",
        marker=dict(color="rgba(0, 200, 200, 0.6)"),
        hovertemplate="Layer: %{x}<br>Max Gradient: %{y:.6f}<extra></extra>",
    )
    trace_mean = go.Bar(
        x=layers,
        y=avg_grads,
        name="Mean Gradient",
        marker=dict(color="rgba(0, 0, 200, 0.6)"),
        hovertemplate="Layer: %{x}<br>Mean Gradient: %{y:.6f}<extra></extra>",
    )
    trace_l2 = go.Scatter(
        x=layers,
        y=l2_norms,
        name="L2 Norm",
        mode="lines+markers",
        line=dict(color="purple", dash="solid"),
        hovertemplate="Layer: %{x}<br>L2 Norm: %{y:.6f}<extra></extra>",
    )

    fig = go.Figure(data=[trace_max, trace_mean, trace_l2])

    fig.update_layout(
        title="Gradient Flow",
        xaxis=dict(
            title="Layers",
            tickangle=45,
            tickmode="array",
            tickvals=layers,
            ticktext=layers,
        ),
        yaxis=dict(title="Gradient Value", autorange=True, nticks=20),
        barmode="overlay",
        hovermode="x unified",
        margin=dict(b=150),
    )

    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
    return html_str
