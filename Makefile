# Makefile for xhisper

CC = gcc
CFLAGS = -O2 -Wall -Wextra
PREFIX = /usr/local
BINDIR = $(PREFIX)/bin

all: xhispertool test

# -ldl: dlopen of libX11 for wait-keys-released (part of libc since glibc 2.34)
xhispertool: xhispertool.c
	$(CC) $(CFLAGS) xhispertool.c -o xhispertool -ldl
	ln -sf xhispertool xhispertoold

test: test.c
	$(CC) $(CFLAGS) test.c -o test

install: xhispertool xhisper.sh xhisper_transcribe.py xhisper_ime.py
	install -d $(DESTDIR)$(BINDIR)
	install -m 755 xhispertool $(DESTDIR)$(BINDIR)/xhispertool
	ln -sf xhispertool $(DESTDIR)$(BINDIR)/xhispertoold
	install -m 755 xhisper.sh $(DESTDIR)$(BINDIR)/xhisper
	install -m 755 xhisper_transcribe.py $(DESTDIR)$(BINDIR)/xhisper_transcribe
	install -m 755 xhisper_ime.py $(DESTDIR)$(BINDIR)/xhisper_ime

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/xhisper
	rm -f $(DESTDIR)$(BINDIR)/xhispertool
	rm -f $(DESTDIR)$(BINDIR)/xhispertoold
	rm -f $(DESTDIR)$(BINDIR)/xhisper_transcribe
	rm -f $(DESTDIR)$(BINDIR)/xhisper_ime

clean:
	rm -f xhispertool xhispertoold test xhisper_transcribe.pyc

.PHONY: all install uninstall clean
