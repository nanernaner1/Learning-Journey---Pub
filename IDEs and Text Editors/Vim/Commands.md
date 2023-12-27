# Commands

## Terminal

### Used to create a terminal shell within vim to prevent closing the Vim session window. (1) benefit to this is you can safely undo/redo changes made to system files and /etc changes without leaving vim.

```
E# After entering vim
:ter
# or
:terminal

## Exiting new terminal window and closing
ctrl + W + w

## To open terminal window 'vertically' instead

:vertical ter
# or
:vert ter

# To quit all vim windows and "Discard" all changes
# First move to non-terminal window with
ctrl + W + w

# Then type
:quitall!
# or
:qa!
```
