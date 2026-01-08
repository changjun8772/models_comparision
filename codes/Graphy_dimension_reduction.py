from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from classi.Features.Smiles_Graphy import Smiles_Graphy

# train_data和test_data分别是训练集和测试集
sg1 = Smiles_Graphy(train_data['Smiles'].values, embedding_dim=256)
sg2 = Smiles_Graphy(test_data['Smiles'].values, embedding_dim=256)
train_graphy_features = sg1.Get_Features().drop(columns=['smiles'])
test_graphy_features = sg2.Get_Features().drop(columns=['smiles'])

rf = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=42)
boruta2 = BorutaPy(rf, n_estimators='auto', alpha=0.05, max_iter=100)
boruta2.fit(train_graphy_features, y_train)
graphy_features_sel = train_graphy_features[:, boruta2.support_]
# boruta2.support_可以保存为csv，方便后续从其它的图特征中获取相同的特征
# 例如： pd.DataFrame({'mask':boruta2.support_}).to_csv("data/graphy_mask.csv", index=False)
