#ifndef __GLUTWRAPPER_H__
#define __GLUTWRAPPER_H__
#include <string>
#include <vector>
#include <map>

class GLUTCallbacks {
public:
	virtual ~GLUTCallbacks() {}
	virtual void display() {}
	virtual void reshape(int width, int height) {}
	virtual void mouse(unsigned buttons, int x, int y) {}
	virtual void keyPressed(unsigned char) {}
	virtual void timer() {}
	virtual void menu(const std::string &elem) {}
};

class GLUTMenuEntry {
public:
	GLUTMenuEntry(const std::string &n): name(n) {}
	void addEntry(GLUTMenuEntry &e) { entries.push_back(e); }
	void addEntry(const std::string &n) { entries.push_back(GLUTMenuEntry(n)); }
	std::string &getName() { return name; }
	unsigned size() { return entries.size(); }
	GLUTMenuEntry & operator[](unsigned i) { return entries[i]; }
private:
	std::string name;
	std::vector<GLUTMenuEntry> entries;
};

class GLUTWrapper {
private:
	GLUTWrapper();
public:
	static GLUTWrapper *create(int *argc, char *argv[]);
	void setTitle(const std::string &);
	void postRedraw();
	void setCallbacks(GLUTCallbacks *);
	void run();
	void exit(int status = 0);
	void startTimer(int);
	void createMenu(GLUTMenuEntry &);
private:
	static void displayFunc();
	static void reshapeFunc(int, int);
	static void mouseMotionFunc(int, int);
	static void mouseFunc(int,int,int,int);
	static void timerFunc(int);
	static void keyboardFunc(unsigned char, int, int);
	static void menuFunc(int);
	int createSubmenu(GLUTMenuEntry &);
	void addMenuEntry(const std::string &name);
	unsigned mouseButtons;
	GLUTCallbacks *callbacks;
	int winId;
	static GLUTWrapper *wrapper;
	std::map<int,std::string> menuEntries;
};

#endif /* __UTILS_H__ */
