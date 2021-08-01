
import os
import shutil
import sqlite3

database = './FaceBase.db'
datasets = './datasets'
def deleteALLinfor():
    dir = os.listdir('E:\desk/pp_relevance/datasets')
    dir_num = len(dir)
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    for i in range(dir_num):
        dir_name = dir[i]
        stu_id = dir_name[4:16]
        cursor.execute('DELETE FROM users WHERE stu_id=?', (stu_id,))
        shutil.rmtree('{}/stu_{}'.format(datasets, stu_id))

    cursor.close()
    conn.commit()
    print("删除成功")
if __name__ == '__main__':
    deleteALLinfor()