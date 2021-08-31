import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import seaborn as sns
from st_aggrid import AgGrid

st.title('Loot Universe')
st.text('How to use?')
with st.expander("See explanantion"):
    st.text("""
    Click on column headings to sort and filter data.
    """)

DATA_URL = "data/loot_updated.parquet"

@st.cache
def load_data(DATA_URL):
    data = pd.read_parquet(DATA_URL)
    return data

df = load_data(DATA_URL)
df_filtered = df[['lootId','score', 'rarest', 'weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring']]
df.style.set_properties(**{'text-align': 'left'}).set_table_styles([ dict(selector='th', props=[('text-align', 'left')] ) ])

st.subheader('Filter Loot')
LootId = st.text_input('Enter loot ID:', 1)
id = df.loc[df['lootId'] == int(LootId)]
id.style.set_properties(**{'text-align': 'left'}).set_table_styles([ dict(selector='th', props=[('text-align', 'left')] ) ])
st.write(id[['lootId','score', 'rarest',]])
col1, col2, col3 = st.columns(3)
col1.write(id[['weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring']].T)
col2.write(id[['weapon_rarity','chest_rarity', 'head_rarity', 'waist_rarity', 'foot_rarity', 'hand_rarity', 'neck_rarity', 'ring_rarity']].T)
col3.write(id[['weapon_count','chest_count', 'head_count', 'waist_count', 'foot_count', 'hand_count', 'neck_count', 'ring_count']].T)

st.subheader('Attribute Filter')

#selection = st.selectbox('Select filter:', ['rarest','weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring'])

#if selection == 'rarest':
rarest = st.slider('How rare?', 1, 8000, (1, 10))
query = df_filtered[df_filtered['rarest'].between(rarest[0], rarest[1])]
# elif selection == 'weapon':
#     weapon = st.selectbox("Weapon", df_filtered['weapon'])
#     query = df_filtered.query("weapon==@weapon")
# elif selection == 'chest':
#     chest = st.selectbox("Chest",df_filtered['chest'])
#     query = df_filtered.query("chest==@chest")
# elif selection == 'head':
#     head = st.selectbox("Head", df_filtered['head'])
#     query = df_filtered.query("head==@head")
# elif selection == 'waist':
#     waist = st.selectbox("Waist",df_filtered['waist'])
#     query = df_filtered.query("waist==@waist")
# elif selection == 'foot':
#     foot = st.selectbox("Foot", df_filtered['foot'])
#     query = df_filtered.query("foot==@foot")
# elif selection == 'hand':
#     hand = st.selectbox("Hand",df_filtered['hand'])
#     query = df_filtered.query("hand==@hand")
# elif selection == 'neck':
#     neck = st.selectbox("Neck", df_filtered['neck'])
#     query = df_filtered.query("neck==@neck")
# elif selection == 'ring':
#     ring = st.selectbox("ring",df_filtered['ring'])
#     query = df_filtered.query("ring==@ring")

st.write(query)
st.subheader('Supercharge filter')
AgGrid(df_filtered)


st.subheader("Loot Relationships")
st.text('Dimensionality reduction and clustering to find relationship among loots')
labels_tsne_scale = df['tsne_clusters']
fig, ax = plt.subplots(figsize = (10,6), dpi=300)
fig.suptitle('Loot clusters', fontsize=20)
sns.scatterplot(df.loc[:,'tsne1'],df.loc[:,'tsne2'],hue=labels_tsne_scale, palette='Set1', s=100, alpha=0.6)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.legend()
fig.tight_layout()
st.pyplot(fig)
st.caption('Loots plotted on 2 dimensions')

cluster_no = range(0, 14)
clusters = st.selectbox("Select cluster", cluster_no)

groups = df.loc[df['tsne_clusters'] == clusters]
col1, col2, col3 = st.columns(3)
col1.metric(label="Group counts:", value=len(groups))
col2.metric("Top loot:", value = str(groups['rarest'].min()))
robe_count = groups.loc[groups['chest'] == 'Divine Robe', 'chest'].count()
col3.metric("No divine robes", value= str(robe_count))





st.text('Relationships open to interpretation.')
AgGrid(groups[['lootId','score', 'rarest','sqdist','weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring']])

selection = st.selectbox('Select equipment:', ['weapon_rarity', 'chest_rarity', 'head_rarity', 'waist_rarity', 'foot_rarity', 'hand_rarity', 'neck_rarity', 'ring_rarity'])
eq = df[selection]
fig, ax = plt.subplots(figsize = (10,6), dpi=300)
fig.suptitle('Loot sub-clusters', fontsize=20)
sns.scatterplot(groups.loc[:,'tsne1'],groups.loc[:,'tsne2'],hue=eq, palette='Set1', s=100, alpha=0.6)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.legend()
fig.tight_layout()
st.pyplot(fig)

st.subheader('Values')
selection = st.selectbox('Select equipment:', ['weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring'])
dft = groups.groupby(f"{selection}_rarity")[selection].value_counts().reset_index(name='Count')
fig, ax = plt.subplots(figsize = (20,8))
plt.title(f"{selection}" + ' distributions')
sns.barplot(x=selection, y='Count', data=dft, hue=f"{selection}_rarity", dodge=False)
plt.legend(title='Item Rarity', bbox_to_anchor=(1, 1), loc='upper right')
plt.xlabel('Item name')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig)



