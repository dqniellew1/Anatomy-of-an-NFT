import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import seaborn as sns
from st_aggrid import AgGrid

st.set_page_config(
     page_title="Loot Universe",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
)

DATA_URL = "data/loot_updated1.parquet"
ADD_URL = "data/address.parquet"

@st.cache
def load_data(DATA_URL):
    data = pd.read_parquet(DATA_URL)
    address = pd.read_parquet(ADD_URL)
    return data, address

st.title('Loot Universe')

PAGES = (
    "Filter tool",
    "Relationships",
    "Attributes sheet"
)

#pd.merge(df, address, left_index=True, right_index=True, left_on='lootId', right_on='loot_id )


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio('Page',PAGES)

    if page == 'Filter tool':
        st.subheader('How to use?')
        st.info("Click on Ξ in the column headings to sort and filter data.")

        df, add = load_data(DATA_URL)
        df_filtered = df[['lootId','score', 'rarest','weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring']]
        st.subheader('Filter Loot')
        LootId = st.text_input('Enter loot ID:', 1)
        id = df.loc[df['lootId'] == int(LootId)]
        st.write(id[['lootId','score', 'rarest']])
        col1, col2, col3 = st.columns([6,4,3])
        col1.write(id[['weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring']].T)
        col2.write(id[['weapon_rarity','chest_rarity', 'head_rarity', 'waist_rarity', 'foot_rarity', 'hand_rarity', 'neck_rarity', 'ring_rarity']].T)
        col3.write(id[['weapon_count','chest_count', 'head_count', 'waist_count', 'foot_count', 'hand_count', 'neck_count', 'ring_count']].T)
        
        st.markdown('##')
        st.subheader("Wallet profiler")
        wallet = pd.merge(df, add, left_index=True, right_index=True, left_on='lootId', right_on='loot_id')
        addID = st.text_input('Enter wallet address:', str('0xC6c41119Af1e0840357245c66baAf0e21B694D4d').lower())
        address = wallet.loc[wallet['address'] == str(f"{addID}").lower()]
        st.write(address[['lootId','score', 'rarest']])
        st.write(address[['weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring']])
        st.write(address[['weapon_rarity','chest_rarity', 'head_rarity', 'waist_rarity', 'foot_rarity', 'hand_rarity', 'neck_rarity', 'ring_rarity']])
        st.write(address[['weapon_count','chest_count', 'head_count', 'waist_count', 'foot_count', 'hand_count', 'neck_count', 'ring_count']])
        
        st.markdown('#')
        st.subheader('Attribute Filter')
        st.subheader('Supercharge filter')
        AgGrid(df_filtered)
        st.markdown('#')

    if page == "Relationships":
        df, _ = load_data(DATA_URL)
        df_filtered = df[['lootId','score', 'rarest','weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring']]

        st.title("Loot Relationships")
        st.text('Dimensionality reduction and clustering to find relationships among loots')
        st.text('Model found 14 loot families.')
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

        st.subheader("Cluster statistics:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Group counts:", value=len(groups))
        col2.metric("Highest rank loot:", value = str(groups['rarest'].min()))
        col3.metric("Lowest rank loot:", value = str(groups['rarest'].max()))
        eq_count = groups[groups['head'].str.contains("Demon")]
        col4.metric("# Demon crowns", value= len(eq_count))

        # Value counts key, value pairs
        wepvalues = groups['weapon'].value_counts().keys().tolist()
        wepcounts = groups['weapon'].value_counts().tolist()

        chestvalues = groups['chest'].value_counts().keys().tolist()
        chestcounts = groups['chest'].value_counts().tolist()

        headvalues = groups['head'].value_counts().keys().tolist()
        headcounts = groups['head'].value_counts().tolist()

        waistvalues = groups['waist'].value_counts().keys().tolist()
        waistcounts = groups['waist'].value_counts().tolist()

        footvalues = groups['foot'].value_counts().keys().tolist()
        footcounts = groups['foot'].value_counts().tolist()

        handvalues = groups['hand'].value_counts().keys().tolist()
        handcounts = groups['hand'].value_counts().tolist()

        neckvalues = groups['neck'].value_counts().keys().tolist()
        neckcounts = groups['neck'].value_counts().tolist()

        ringvalues = groups['ring'].value_counts().keys().tolist()
        ringcounts = groups['ring'].value_counts().tolist()

        col21, col22 = st.columns([2,1])
        col23, col24 = st.columns([2,1])
        col25, col26 = st.columns([2,1])
        col27, col28 = st.columns([2,1])

        col21.text(str('Most common weapon: ' + str(str(wepvalues[0]) + ': ' + str(wepcounts[0]))))
        col22.text(str('Most common chest: ' + str(str(chestvalues[0]) + ': ' + str(chestcounts[0]))))
        col23.text(str("Most common head: " + str(str(headvalues[0]) + ': ' + str(headcounts[0]))))
        col24.text(str("Most common waist: " + str(str(waistvalues[0]) + ': ' + str(waistcounts[0]))))
        col25.text(str("Most common foot: " + str(str(footvalues[0]) + ': ' + str(footcounts[0]))))
        col26.text(str("Most common hand: " + str(str(handvalues[0]) + ': ' + str(handcounts[0]))))
        col27.text(str("Most common neck: " + str(str(neckvalues[0]) + ': ' + str(neckcounts[0]))))
        col28.text(str("Most common ring: " + str(str(ringvalues[0]) + ': ' + str(ringcounts[0]))))
        st.markdown('##')
        st.write('--')
        st.text('The relationships between each loot are open to interpretations.')
        AgGrid(groups[['lootId','score', 'rarest','sqdist','weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring']])

        st.markdown('##')
        selection = st.selectbox('Select equipment:', ['weapon_rarity', 'chest_rarity', 'head_rarity', 'waist_rarity', 'foot_rarity', 'hand_rarity', 'neck_rarity', 'ring_rarity'])
        st.text("Look at different categories in each cluster.")
        eq = df[selection]
        fig, ax = plt.subplots(figsize = (10,6), dpi=300)
        fig.suptitle('Loot sub-clusters', fontsize=20)
        sns.scatterplot(groups.loc[:,'tsne1'],groups.loc[:,'tsne2'],hue=eq, palette='Set1', s=100, alpha=0.6)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.legend()
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown('##')
        selection = st.selectbox('Select equipment:', ['weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring'])
        st.text("Look at distributions in each cluster.")
        dft = groups.groupby(f"{selection}_rarity")[selection].value_counts().reset_index(name='Count')
        fig, ax = plt.subplots(figsize = (20,8))
        plt.title(f"{selection}" + ' distributions')
        sns.barplot(x=selection, y='Count', data=dft, hue=f"{selection}_rarity", dodge=False)
        plt.legend(title='Item Rarity', bbox_to_anchor=(1, 1), loc='upper right')
        plt.xlabel('Item name')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()
        st.pyplot(fig)
    
    if page == 'Attributes sheet':
        st.title('Attibutes sheet')
        df, _ = load_data(DATA_URL)
        df_filtered = df[['lootId','score', 'rarest', 'weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring']]        
        selection = st.selectbox('Select equipment:', ['weapon', 'chest', 'head', 'waist', 'foot', 'hand', 'neck', 'ring'])
        st.text('Common items appear 375 or more times.')
        st.text('Uncommon items appear less than 375 times.')
        st.text('Rare items appear less than 358 times.')
        st.text('Epic items appear less than 101 times.')
        st.text('Legendary items appear less than 10 times.')
        st.text('Mythic items appear exactly 1 time.')
        col1, col2 = st.columns([3,2])
        col1.dataframe(df.groupby(f"{selection}_rarity")[selection].value_counts())
        col2.write(df[f"{selection}_rarity"].value_counts())



if __name__ == "__main__":
    main()