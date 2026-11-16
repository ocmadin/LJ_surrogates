import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from LJ_surrogates.analysis.analysis import SimulatedDataset

class ComponentOrderTests(unittest.TestCase):
    def dataset(self, df):
        obj=SimulatedDataset.__new__(SimulatedDataset)
        obj.df=df
        return obj

    def test_binary_permutation_and_pure_policy(self):
        ketone='CC(=O)CC(C)C';hexanol='CCCCCCO'
        rows=[(1,'CCO',None),(1,ketone,None),(1,hexanol,None),
              (2,'CCO',ketone),(2,ketone,'CCO'),(2,'CCO',hexanol),
              (2,hexanol,'CCO'),(2,'CCO','CCC'),(2,'CCC','CCO')]
        for dtype in ['object','string','category']:
            with self.subTest(dtype=dtype):
                df=pd.DataFrame(rows,columns=['N Components','Component 1','Component 2'],index=[0,0,1,1,2,2,3,3,4])
                for col in ['Component 1','Component 2']:df[col]=df[col].astype(dtype)
                before=df.copy(deep=True); obj=self.dataset(df)
                obj.remove_problematic_points()
                assert_frame_equal(obj.df,df.iloc[[0,2,7,8]])
                assert_frame_equal(df,before)
                obj.remove_problematic_points()
                assert_frame_equal(obj.df,df.iloc[[0,2,7,8]])

    def test_empty(self):
        df=pd.DataFrame({'N Components':pd.Series(dtype=int),'Component 1':pd.Series(dtype=object),'Component 2':pd.Series(dtype=object)})
        obj=self.dataset(df);obj.remove_problematic_points();assert_frame_equal(obj.df,df)

if __name__=='__main__': unittest.main()
