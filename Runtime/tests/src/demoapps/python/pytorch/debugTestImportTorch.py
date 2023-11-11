#@author: Steven Tang <steven.tang@bytedance.com>

import gdb
import threading
import time

class MyBreakpoint1(gdb.Breakpoint):
    def stop(self) -> bool:
        #print('Here')
        #t=threading.Thread(target=prettyPrinting)
        #t.start()
        return True

gdb.Breakpoint('dlopen')
gdb.execute('layout split')
gdb.execute('r  TestImportTorch.py > a.out 2> a.err')
gdb.execute('up')
dlOpenContent=gdb.execute('p/x *((int**)0x710af0)',to_string=True)
gdb.execute('x/3i %s'%(dlOpenContent.split()[-1]))
