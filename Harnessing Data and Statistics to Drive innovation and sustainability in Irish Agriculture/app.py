#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

#######################
# Page configuration
st.set_page_config(
    page_title="European Agriculture and Framing",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


df = pd.read_csv('df.csv')


#######################
# Sidebar
with st.sidebar:
    st.title('European Agriculture Dashboard')
    
    # Dropdown to select the column
    column_list = [col for col in df.columns if col not in ['country_code', 'area', 'year','Unnamed: 0.3','Unnamed: 0.2','Unnamed: 0.4','Unnamed: 0.1','Unnamed: 0','']]  # Exclude non-relevant columns
    selected_column = st.selectbox('Select a column to display', column_list, index=0)

    year_list = list(df.year.unique())[::-1]
    
    selected_year = st.selectbox('Select a year', year_list)
    df_selected_year = df[df.year == selected_year]
    df_selected_year_sorted = df_selected_year.sort_values(by="agriculture_value", ascending=False)

#######################
# Plots

st.header('European Agriculture Dashboard')

# Heatmap
def make_heatmap(df, year):
    # Create the heatmap
    heatmap = alt.Chart(df).mark_rect().encode(
        y=alt.Y('year:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
        x=alt.X('area:O', axis=alt.Axis(title="Area", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=alt.Color('agriculture_value:Q',
                        legend=None,
                        scale=alt.Scale(scheme='viridis')),  # Default color scheme
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
    ).properties(
        width=900
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    )

    return heatmap


# Choropleth map
def make_choropleth(df, column, title='Choropleth Map'):
    fig = px.choropleth(
        df,
        locations='country_code',  # Country codes for geographic mapping
        color=column,  # Use the selected column for color scale
        hover_name="area",  # Country/area names
        color_continuous_scale=px.colors.sequential.Plasma  # Color scale
    )

    # Set geographic scope to Europe and adjust layout
    fig.update_layout(
        geo=dict(
            scope='europe',  # Focus on Europe
            projection_scale=1.5,  # Zoom in
            center={"lat": 55, "lon": 10},  # Center the map on Europe
        ),
        width=1000,  # Increase figure width
        height=800,  # Increase figure height
        title_font=dict(size=20),  # Adjust title font size
        title_text=title  # Add title dynamically
    )

    return fig


# Function to create a bar chart
def make_bar_chart(df, select_column, title="Top Regions by Metric"):
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f'{select_column}:Q', title=select_column.capitalize()),
        y=alt.Y('area:N', sort='-x', title="Region"),
        color=alt.Color(f'{select_column}:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['area', f'{select_column}']
    ).properties(
        title=title,
        width=800,
        height=600
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16
    )
    return chart

#######################
# Dashboard Main Panel


col = st.columns((2, 4, 2), gap='medium')


st.markdown('#### Country Metrics Change')

# Select a metric for analysis
selected_metric = st.selectbox('Select a Metric', 
                               [col for col in df.columns if df[col].dtype in [float, int] and col not in ['year', 'country_code']])


with col[0]:
    st.markdown(f'#### Top Regions by {selected_column}')
    bar_chart = make_bar_chart(df_selected_year.head(10), selected_column, title=f"Top Regions by {selected_column}")
    st.altair_chart(bar_chart, use_container_width=True)

# Choropleth
with col[1]:
    st.markdown(' Distribution Across Regions')
    choropleth = make_choropleth(df_selected_year, selected_column, title=f'{selected_column} in {selected_year}')
    st.plotly_chart(choropleth, use_container_width=True)

    heatmap = make_heatmap(df, df_selected_year)
    st.altair_chart(heatmap, use_container_width=True)


with col[2]:
    st.markdown('#### Top Countries agriculture streamlit run app.pyoutput by year')

    st.dataframe(df_selected_year_sorted,
                 column_order=("area", "agriculture_value"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "area": st.column_config.TextColumn(
                        "area",
                    ),
                    "Agriculture value": st.column_config.ProgressColumn(
                        "agriculture value",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_year_sorted.agriculture_value),
                     )}
                 )


with st.expander('About', expanded=True):
        st.write('''
            - Data: [ https://www.fao.org/faostat/en/#data/QCL ].
            - Data: [ https://ourworldindata.org/ ]''')