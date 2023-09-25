import tkinter as tk

window = tk.Tk() # 기본 윈도우 생성

window.title("Hello")
window.geometry("640x400+100+100") #창의 너비x높이+x좌표+y좌표 지정
window.resizable(False, False) #창의 상하, 좌우 조절 지정

window.mainloop()
