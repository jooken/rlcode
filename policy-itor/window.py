import tkinter

window = tkinter.Tk() # 기본 윈도우 생성

window.title("Hello")
window.geometry("640x400+100+100") #창의 너비x높이+x좌표+y좌표 지정
window.resizable(False, False) #창의 상하, 좌우 조절 지정

# 레이블의 크기, 글자색, 테두리 스타일 지정하기
label=tkinter.Label(window, text="Python", width=10, height=5, fg="red", relief="sunken")
label.pack()

window.mainloop()
